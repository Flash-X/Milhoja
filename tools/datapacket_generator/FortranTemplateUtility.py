
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
    def iterate_externals(cls, connectors, size_connectors, externals):
        ...
        # nTiles_value =  f'// Check for overflow first to avoid UB\n' + \
        #                 f'// TODO: Should casting be checked here or in base class?\n' + \
        #                 f'if (tiles_.size() > INT_MAX)\n\tthrow std::overflow_error("[_param:class_name pack] nTiles was too large for int.");\n' + \
        #                 f'_nTiles_h = static_cast<{externals["nTiles"]["type"]}>(tiles_.size());'
        
    
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
    ):
        ...

    @classmethod
    def iterate_tile_in_out(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tileinout: OrderedDict, 
    ):
        ...

    @classmethod
    def iterate_tile_out(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tileout: OrderedDict, 
    ):
        ...

    @classmethod
    def iterate_tile_scratch(
        cls,
        connectors: dict, 
        size_connectors: dict, 
        tilein: OrderedDict, 
    ):
        ...