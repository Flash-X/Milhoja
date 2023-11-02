
from collections import OrderedDict
from DataPacketMemberVars import DataPacketMemberVars
from TemplateUtility import TemplateUtility

class FortranTemplateUtility(TemplateUtility):
    FARRAY_MAPPING = {
        "int": "IntVect",
        "real": "RealVect"
    }

    F_HOST_EQUIVALENT = {
        'RealVect': 'real',
        'IntVect': 'int'
    }

    @classmethod
    def iterate_externals(cls, connectors: dict, size_connectors: dict, externals: OrderedDict):
        """
        Iterates the external variables section 
        and adds the necessary connectors.

        :param dict connectors: The dictionary containing all connectors for use with CGKit.
        :param dict size_connectors: The dictionary containing all size connectors for determining the sizes of each item in the packet.
        :param OrderedDict constructor: The dictionary containing the data for the ptv section in the DataPacket JSON.
        :rtype: None
        """
        cls.section_creation(cls._EXT, externals, connectors, size_connectors)
        connectors[cls._HOST_MEMBERS] = []
        # we need to set nTiles value here as a link, _param does not work as expected.
        
        nTiles_value =  f'// Check for overflow first to avoid UB\n' + \
                        f'// TODO: Should casting be checked here or in base class?\n' + \
                        f'if (tiles_.size() > INT_MAX)\n\tthrow std::overflow_error("[_param:class_name pack] nTiles was too large for int.");\n' + \
                        f'_nTiles_h = static_cast<{externals["nTiles"]["type"]}>(tiles_.size());'
        connectors[cls._NTILES_VALUE] = [nTiles_value]

        # # MOVE THROUGH EVERY CONSTRUCTOR ITEM
        for key,var_data in externals.items():
            size_equation = f'sizeof({var_data["type"]})'
            if var_data["extents"]:
                size_equation = f'{size_equation} * {" * ".join(var_data["extents"])}'
            info = DataPacketMemberVars(
                item=key, dtype=var_data["type"], 
                size_eq=size_equation, per_tile=False
            )

            # nTiles is a special case here. nTiles should not be included 
            # in the constructor, and it has its own host variable generation.
            if key != 'nTiles':
                connectors[cls._CON_ARGS].append(f'{info.dtype} {key}')
                connectors[cls._HOST_MEMBERS].append(info.host)
            # add the necessary connectors for the constructor section.
            connectors[cls._PUB_MEMBERS].extend([
                f'{info.dtype} {info.host};\n',
                f'{info.dtype}* {info.device};\n'
            ])
            
            set_host = f'{{{key}}}'
            # NOTE: it doesn't matter what we set nTiles to here.
            #       nTiles always gets set in pack, and we cannot set nTiles in here using tiles_
            #       because tiles_ has not been filled at the time of this packet's construction.
            if key == "nTiles":
                set_host = '{0}'
            
            connectors[cls._SET_MEMBERS].extend([
                f'{info.host}{set_host}', 
                f'{info.device}{{nullptr}}'
            ])
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
            )
            cls.set_pointer_determination(connectors, cls._EXT, info)
            connectors[f'memcpy_{cls._EXT}'].append(
                f'std::memcpy({info.pinned}, static_cast<void*>(&{info.host}), {info.size});\n'
            )
    
    @classmethod
    def iterate_tilemetadata(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tilemetadata: OrderedDict,
        num_arrays: int
    ):
        """
        Iterates the tilemetadata section of the JSON.
        
        :param dict connectors: The dict containing all connectors for cgkit.
        :param dict size_connectors: The dict containing all size connectors for variable sizes.
        :param OrderedDict tilemetadata: The dict containing information from the tile-metadata section in the JSON.
        :param str language: The language to use
        :param int num_arrays: The number of arrays inside tile-in, tile-in-out, tile-out, and tile-scratch.
        """

        cls.section_creation(cls._T_MDATA, tilemetadata, connectors, size_connectors)
        connectors[cls._T_DESCRIPTOR] = []

        for item,data in tilemetadata.items():
            source = data['source'].replace('tile_', '')
            item_type = data['type']
            size_eq = f"MILHOJA_MDIM * sizeof({item_type})"
            info = DataPacketMemberVars(item=item, dtype=item_type, size_eq=size_eq, per_tile=True)

            # extend each connector
            connectors[cls._PUB_MEMBERS].extend( [f'{item_type}* {info.device};\n'] )
            connectors[cls._SET_MEMBERS].extend( [f'{info.device}{{nullptr}}'] )
            connectors[cls._SIZE_DET].append( f'static constexpr std::size_t {info.size} = {size_eq};\n' )
            cls.set_pointer_determination(connectors, cls._T_MDATA, info)

            # data type depends on the language. If there exists a mapping for a fortran data type for the given item_type,
            # use that instead.
            info.dtype = cls.FARRAY_MAPPING.get(item_type, item_type)
            connectors[cls._T_DESCRIPTOR].append(
                f'const auto {source} = tileDesc_h->{source}();\n'
            )

            # unsigned does not matter in fortran.
            # TODO: This might not work completely when needing an unsigned int 
            #       as a variable type. Try generating spark packets to resolve this.
            info.dtype = info.dtype.replace('unsigned', '')
            # if the language is fortran and there exists a fortran data type equivalent (eg, IntVect -> int array.)
            use_ref = ""
            if info.dtype in cls.F_HOST_EQUIVALENT:
                fix_index = '+1' if info.dtype == str('IntVect') else '' # indices are 1 based, so bound arrays need to adjust
                info.dtype = cls.F_HOST_EQUIVALENT[info.dtype]
                construct_host = f"[MILHOJA_MDIM] = {{ {source}.I(){fix_index}, {source}.J(){fix_index}, {source}.K(){fix_index} }}"
                use_ref = "" # don't need to pass by reference with primitive arrays
            else:
                construct_host = f' = static_cast<{item_type}>({info.host})'
                use_ref = "&" # need a reference for Vect objects.
            cls.tile_metadata_memcpy(connectors, construct_host, use_ref, info)

    @classmethod
    def tile_metadata_memcpy(cls, connectors: dict, construct: str, use_ref: str, info: DataPacketMemberVars):
        """
        Adds the memcpy portion for the metadata section in a fortran packet.
        
        :param dict connectors: The dictionary containing all cgkit connectors.
        :param str construct: The generated host variable to copy to the pinned pointer location
        :param str use_ref: Use a reference to the host item.
        :param DataPacketMemberVars info: Contains information for formatting the name to get variable names.
        """
        connectors[f'memcpy_{cls._T_MDATA}'].extend([
            f'{info.dtype} {info.host}{construct};\n',
            f'char_ptr = static_cast<char*>(static_cast<void*>({info.pinned})) + n * {info.size};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({use_ref}{info.host}), {info.size});\n\n',
        ])

    @classmethod
    def iterate_tile_in(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tilein: OrderedDict, 
    ) -> None:
        """
        Iterates the tile in section of the JSON.
        
        :param dict connectors: The dict containing all connectors for cgkit.
        :param dict size_connectors: The dict containing all size connectors for items in the data packet.
        :param OrderedDict tilein: The dict containing the information in the tile_in section.
        :param str language: The language of the corresponding task function.
        """
        cls.section_creation(cls._T_IN, tilein, connectors, size_connectors)
        for item,data in tilein.items():
            # gather all information from tile_in section.
            extents = data['extents']
            mask_in = data['variables_in']
            dtype = data['type']

            extents = ' * '.join(f'({item})' for item in extents)
            unks = f'{mask_in[1]} - {mask_in[0]} + 1'
            info = DataPacketMemberVars(
                item=item, dtype=dtype, 
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})', per_tile=True
            )
            
            # Add necessary connectors.
            connectors[cls._PUB_MEMBERS].append(
                f'{info.dtype}* {info.device};\n'
                f'{info.dtype}* {info.pinned};\n'
            )
            connectors[cls._SET_MEMBERS].extend(
                [
                    f'{info.device}{{nullptr}}',
                    f'{info.pinned}{{nullptr}}'
                ]
            )
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
            )
            cls.set_pointer_determination(connectors, cls._T_IN, info, True)
            cls.add_memcpy_connector(
                connectors, cls._T_IN, 
                extents, item, 
                mask_in[0], mask_in[1], 
                info.size, info.dtype, data['structure_index'][0])

    @classmethod
    def iterate_tile_in_out(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tileinout: OrderedDict
    ):
        """
        Iterates the tileinout section of the JSON.

        :param dict connectors: The dict containing all connectors for use with cgkit.
        :param dict size_connectors: The dict containing all size connectors for items in the JSON.
        :param OrderedDict tileinout: The dict containing the data from the tile-in-out section of the datapacket json.
        :param str language: The language to use.
        """
        cls.section_creation(cls._T_IN_OUT, tileinout, connectors, size_connectors)
        connectors[f'memcpy_{cls._T_IN_OUT}'] = []
        connectors[f'unpack_{cls._T_IN_OUT}'] = []
        # unpack all items in tile_in_out
        for item,data in tileinout.items():
            in_mask = data['variables_in']
            out_mask = data['variables_out']
            dtype = data['type']
            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            unks = f'{in_mask[1]} - {in_mask[0]} + 1'
            info = DataPacketMemberVars(
                item=item, dtype=dtype, 
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})', 
                per_tile=True
            )

            # set connectors
            connectors[cls._PUB_MEMBERS].append(
                f'{info.dtype}* {info.device};\n'
                f'{info.dtype}* {info.pinned};\n'
            )
            connectors[cls._SET_MEMBERS].extend(
                [
                    f'{info.device}{{nullptr}}',
                    f'{info.pinned}{{nullptr}}'
                ]
            )
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
            )
            cls.set_pointer_determination(connectors, cls._T_IN_OUT, info, True)
            cls.add_memcpy_connector(
                connectors, cls._T_IN_OUT, 
                extents, item, in_mask[0], 
                in_mask[1], info.size, info.dtype,
                data['structure_index'][0]
            )
            # here we pass in item twice because tile_in_out pointers get packed and unpacked from the same location.
            cls.add_unpack_connector(
                connectors, cls._T_IN_OUT, 
                extents, out_mask[0], out_mask[1], 
                info.dtype, item,
                data['structure_index'][0]
            )

    @classmethod
    def iterate_tile_out(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tileout: OrderedDict
    ):
        """
        Iterates the tileout section of the JSON.

        TODO: Any pinned pointer that needs to be a datapacket member variable should be a private variable.
        
        :param dict connectors: The dict containing all connectors for use with cgkit.
        :param dict size_connectors: The dict containing all size connectors for items in the JSON.
        :param OrderedDict tileout: The dict containing information from the tile-out section of the data packet JSON.
        :param str language: The language to use. 
        """
        cls.section_creation(cls._T_OUT, tileout, connectors, size_connectors)
        connectors[f'unpack_{cls._T_OUT}'] = []
        for item,data in tileout.items():
            # ge tile_out information
            out_mask = data['variables_out']
            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            dtype = data['type']
            info = DataPacketMemberVars(
                item=item, 
                dtype=dtype, 
                size_eq=f'{extents} * ( {out_mask[1]} - {out_mask[0]} + 1 ) * sizeof({dtype})', 
                per_tile=True
            )

            connectors[cls._PUB_MEMBERS].append(
                f'{info.dtype}* {info.device};\n'
                f'{info.dtype}* {info.pinned};\n'
            )
            connectors[cls._SET_MEMBERS].extend(
                [f'{info.device}{{nullptr}}',
                f'{info.pinned}{{nullptr}}']
            )
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
            )
            cls.set_pointer_determination(connectors, cls._T_OUT, info, True)
            cls.add_unpack_connector(
                connectors, cls._T_OUT, extents, out_mask[0], 
                out_mask[1], info.dtype, info.ITEM,
                data['structure_index'][0]
            )

    @classmethod
    def iterate_tile_scratch(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tilescratch: OrderedDict, 
    ):
        """
        Iterates the tilescratch section of the JSON.
        
        :param dict connectors: The dict containing all connectors for use with cgkit.
        :param dict size_connectors: The dict containing all size connectors for variable sizes.
        :param OrderedDict tilescratch: The dict containing information from the tilescratch section of the JSON.
        :param str language: The language to use when generating the packet. 
        """
        cls.section_creation(cls._T_SCRATCH, tilescratch, connectors, size_connectors)
        for item,data in tilescratch.items():
            exts = data[cls._EXTENTS]
            dtype = data["type"]
            # need to fill farrays with default args
            extents4d = ["1"] * 4
            for idx,val in enumerate(exts):
                extents4d[idx] = str(val)

            extents = ' * '.join(f'({val})' for val in extents4d)
            info = DataPacketMemberVars(
                item=item, dtype=dtype, 
                size_eq=f'{extents} * sizeof({dtype})', 
                per_tile=True
            )

            connectors[cls._PUB_MEMBERS].append( f'{info.dtype}* {info.device};\n' )
            connectors[cls._SET_MEMBERS].append( f'{info.device}{{nullptr}}' )
            connectors[cls._SIZE_DET].append( f'static constexpr std::size_t {info.size} = {info.SIZE_EQ};\n' )
            connectors[f'pointers_{cls._T_SCRATCH}'].append(
                f"""{info.device} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
                f"""ptr_d += {info.total_size};\n\n"""
            )