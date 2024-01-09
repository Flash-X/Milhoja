from collections import OrderedDict
from .DataPacketMemberVars import DataPacketMemberVars
from .TemplateUtility import TemplateUtility


class CppTemplateUtility(TemplateUtility):
    """
    Internal utility class for the DataPacketGenerator for creating packets
    for use with a C++ task function.
    """

    @classmethod
    def iterate_externals(
        cls, connectors: dict, size_connectors: dict, externals: OrderedDict
    ):
        """
        Iterates the external variables section
        and adds the necessary connectors.

        :param dict connectors: The dictionary containing all connectors for
                                use with CGKit.
        :param dict size_connectors: The dictionary containing all size
                                     connectors for determining the sizes of
                                     each item in the packet.
        :param OrderedDict externals: The dictionary containing the data for
                                        the externals section in the TF JSON
        :rtype: None
        """
        cls.section_creation(cls._EXT, externals, connectors, size_connectors)
        connectors[cls._HOST_MEMBERS] = []
        # we need to set nTiles value here as a link, _param does not work as
        # expected.

        nTiles_value = '_nTiles_h = tiles_.size();'
        connectors[cls._NTILES_VALUE] = [nTiles_value]
        cls._common_iterate_externals(connectors, externals)

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
        :param dict size_connectors: The dict containing all size connectors
                                     for variable sizes.
        :param OrderedDict tilemetadata: The dict containing information from
                                         the tile-metadata section in the JSON
        :param int num_arrays: The number of arrays inside tile_in,
                               tile_in_out, tile_out, and tile_scratch.
        """

        cls.section_creation(
            cls._T_MDATA, tilemetadata, connectors, size_connectors
        )
        connectors[cls._T_DESCRIPTOR] = []
        connectors[cls._SIZE_DET].append(
            "static constexpr std::size_t SIZE_FARRAY4D = sizeof(FArray4D);\n"
        )
        cls.insert_farray_size(size_connectors, num_arrays)

        for item, data in tilemetadata.items():
            source = data['source']

            # ..todo:: Temporary fix for generating flash-x packet
            if source == "tile_lbound":
                source = "tile_loGC"
            elif source == "tile_ubound":
                source = "tile_hiGC"

            source = source.replace('tile_', '')
            item_type = data['type']
            size_eq = f"sizeof({item_type})"
            info = DataPacketMemberVars(
                item=item, dtype=item_type, size_eq=size_eq, per_tile=True
            )

            # extend each connector
            connectors[cls._PUB_MEMBERS].extend(
                [f'{item_type}* {info.device};\n']
            )
            connectors[cls._SET_MEMBERS].extend(
                [f'{info.device}{{nullptr}}']
            )
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {size_eq};\n'
            )
            cls.set_pointer_determination(connectors, cls._T_MDATA, info)

            connectors[cls._T_DESCRIPTOR].append(
                f'const auto {source} = tileDesc_h->{source}();\n'
            )

            # TODO: This might not work completely when needing an unsigned
            #       int as a variable type. Try generating spark packets to
            #       resolve this.
            cls.tile_metadata_memcpy(connectors, info, source)

    @classmethod
    def iterate_tile_in(
        cls,
        connectors: dict,
        size_connectors: dict,
        tilein: OrderedDict,
    ):
        """
        Iterates the tile in section of the JSON.

        :param dict connectors: The dict containing all connectors for cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for items in the data packet.
        :param OrderedDict tilein: The dict containing the information in the
                                   tile_in section.
        """
        cls.section_creation(cls._T_IN, tilein, connectors, size_connectors)
        for item, data in tilein.items():
            # gather all information from tile_in section.
            extents = data['extents']
            mask_in = data['variables_in']

            # for now just assume that all index spaces are 1 based
            # since all arrays in the packet are 1 based.
            index_offset = cls.DEFAULT_INDEX_SPACE
            array_size = cls.get_array_size(mask_in, [])

            dtype = data['type']
            extents = ' * '.join(f'({item})' for item in extents)
            unks = f'{str(array_size)} + 1 - {str(index_offset)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_data(
                data, connectors, info, extents, mask_in, None, cls._T_IN
            )

            cls.insert_farray_memcpy(
                connectors, info.ITEM, 'tileDesc_h->loGC()',
                'tileDesc_h->hiGC()', unks, info.dtype
            )

    @classmethod
    def iterate_tile_in_out(
        cls,
        connectors: dict,
        size_connectors: dict,
        tileinout: OrderedDict
    ):
        """
        Iterates the tileinout section of the JSON.

        :param dict connectors: The dict containing all connectors for use
                                with cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for items in the JSON.
        :param OrderedDict tileinout: The dict containing the data from the
                                      tile-in-out section of the datapacket
                                      json.
        """
        cls.section_creation(
            cls._T_IN_OUT, tileinout, connectors, size_connectors
        )
        connectors[f'memcpy_{cls._T_IN_OUT}'] = []
        connectors[f'unpack_{cls._T_IN_OUT}'] = []
        # unpack all items in tile_in_out
        for item, data in tileinout.items():
            in_mask = data['variables_in']
            out_mask = data['variables_out']
            array_size = cls.get_array_size(in_mask, out_mask)
            index_space = cls.DEFAULT_INDEX_SPACE

            dtype = data['type']
            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            unks = f'{array_size} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_data(
                data, connectors, info, extents,
                in_mask, out_mask, cls._T_IN_OUT
            )

            cls.insert_farray_memcpy(
                connectors, item,
                "tileDesc_h->loGC()", "tileDesc_h->hiGC()",
                unks, info.dtype
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

        ..todo::
            * Any pinned pointer that needs to be a datapacket member variable
              should be a private variable.

        :param dict connectors: The dict containing all connectors for use w/
                                cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                    for items in the JSON.
        :param OrderedDict tileout: The dict containing information from the
                                    tile-out section of the data packet JSON.
        """
        cls.section_creation(cls._T_OUT, tileout, connectors, size_connectors)
        connectors[f'unpack_{cls._T_OUT}'] = []
        for item, data in tileout.items():
            # ge tile_out information
            out_mask = data['variables_out']
            array_size = cls.get_array_size([], out_mask)
            index_space = cls.DEFAULT_INDEX_SPACE
            unks = f"{str(array_size)} + 1 - {str(index_space)}"

            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            dtype = data['type']
            info = DataPacketMemberVars(
                item=item,
                dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_data(
                data, connectors, info, extents, None, out_mask, cls._T_OUT
            )

            cls.insert_farray_memcpy(
                connectors,
                item,
                'tileDesc_h->loGC()',
                'tileDesc_h->hiGC()',
                unks,
                info.dtype
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

        :param dict connectors: The dict containing all connectors for use
                                with cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for variable sizes.
        :param OrderedDict tilescratchs: The dict containing information from
                                         the tilescratch section of the JSON.
        :param str language: The language to use when generating the packet.
        """
        cls.section_creation(
            cls._T_SCRATCH, tilescratch, connectors, size_connectors
        )
        for item, data in tilescratch.items():
            lbound = data[cls._LBOUND]
            exts = data[cls._EXTENTS]
            dtype = data["type"]
            # need to fill farrays with default args
            extents4d = ["1"] * 4
            for idx, val in enumerate(exts):
                extents4d[idx] = str(val)

            extents = ' * '.join(f'({val})' for val in extents4d)
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_scratch(
                connectors, info
            )
            # we don't insert into memcpy or unpack because the scratch is
            # only used in the device memory and does not come back.

            # ..todo:
            #    * Does this work fine if we allow anything to be scratch?
            #    * this is not an issue for Fortran TF packets, but the
            #      current
            #      FArray4D implementation for C++ packets seems limiting for
            #      scratch arrays.
            #    * auxC scratch uses loGC and hiGC. How to incorporate this
            #      with
            #      other scratch arrays?
            #    * Change data type of scratch array based on length of
            #      extents in C++ packet.
            lo = f"IntVect{{ LIST_NDIM({', '.join(lbound[:-1])}) }}" \
                if len(exts) == len(lbound) else lbound[0]
            hi = f'({lo}) +' \
                f'( IntVect{{ LIST_NDIM({",".join(extents4d[:-1])}) }} )'
            unks = extents4d[-1]
            cls.insert_farray_memcpy(
                connectors, item, lo, hi, str(unks), info.dtype
            )

    @classmethod
    def insert_farray_size(cls, connectors, num_arrays):
        """Inserts the total size needed to store all farray pointers."""
        line = connectors[f'size_{cls._T_MDATA}']
        insert_index = line.find('(')
        connectors[f'size_{cls._T_MDATA}'] = \
            f'{line[:insert_index + 1]}({num_arrays} * SIZE_FARRAY4D) + ' \
            f'{line[insert_index + 1:]}'

    @classmethod
    def insert_farray_memcpy(
        cls, connectors, item: str, lo: str,
        hi: str, unks: str, data_type: str
    ):
        """
        Insers the farray memcpy and data pointer sections into the data
        packet connectors.

        :param dict connectors: The dict that stores all cgkit connectors.
        :param str item: The item to be stored.
        :param str lo: The low bound of the array
        :param str hi: The high bound of the array
        :param str unks: The number of unknown variables in *item*
        :param str data_type: The data type of *item*
        """
        # create metadata pointers for the f4array classes.
        # ..todo::
        #      * insert 4D/3D/2D/1D arrays as necessary.
        connectors[f'pointers_{cls._T_MDATA}'].append(
            f'FArray4D* _f4_{item}_p = static_cast<FArray4D*>( '
            'static_cast<void*>( ptr_p ) );\n'
            f'_f4_{item}_d = static_cast<FArray4D*>( '
            'static_cast<void*>( ptr_d ) );\n'
            'ptr_p += _nTiles_h * SIZE_FARRAY4D;\n'
            'ptr_d += _nTiles_h * SIZE_FARRAY4D;\n\n'
        )
        # Does not matter what memcpy section we insert into,
        # so we default to T_IN.
        connectors[f'memcpy_{cls._T_IN}'].extend([
            f'FArray4D {item}_device{{ static_cast<{data_type}*>( '
            'static_cast<void*>( static_cast<char*>( '
            f'static_cast<void*>(_{item}_d) ) '
            f'+ n * SIZE_{item.upper()})), {lo}, {hi}, {unks}}};\n',
            'char_ptr = static_cast<char*>( '
            f'static_cast<void*>(_f4_{item}_p) ) + n * SIZE_FARRAY4D;\n',
            'std::memcpy(static_cast<void*>(char_ptr), '
            f'static_cast<void*>(&{item}_device), SIZE_FARRAY4D);\n\n'
        ])

    @classmethod
    # can probably shrink this function and insert it into each data section.
    def insert_farray_information(cls, connectors, tile_data: list):
        """
        Inserts farray items into the data packet.

        :param dict data: The dict containing information from the data
                          packet JSON.
        :param dict connectors: The dict containing all cgkit connectors.
        :param str section: The connectors section to extend.
        """
        # Get all items in each data array.
        # we need to make an farray object for every possible data array
        farrays = {item: sect[item] for sect in tile_data for item in sect}
        # TODO: Use DataPacketMemberVars class for this.
        connectors[cls._PUB_MEMBERS].extend(
            [f'FArray4D* _f4_{item}_d;\n' for item in farrays]
        )
        connectors[cls._SET_MEMBERS].extend(
            [f'_f4_{item}_d{{nullptr}}' for item in farrays]
        )

    @classmethod
    def tile_metadata_memcpy(
        cls,
        connectors,
        info: DataPacketMemberVars,
        alt_name: str
    ):
        """
        The cpp version for the metadata memcpy section.
        :param dict connectors: The dictionary containing all connectors for
                                CGKit.
        :param DataPacketMemberVars info: The information of the DataPacket
                                          variable.
        :param str alt_name: The name of the source pointer to be copied in.
        """
        # Inserts the memcpy portion for tile metadata.
        # Various arguments are unused to share a function call with another
        # func.
        connectors[f'memcpy_{cls._T_MDATA}'].extend([
            f"""char_ptr = static_cast<char*>( static_cast<void*>( """
            f"""{info.pinned} ) ) + n * {info.size};\n""",
            f"""std::memcpy(static_cast<void*>(char_ptr), """
            f"""static_cast<const void*>(&{alt_name}), {info.size});\n\n"""
        ])