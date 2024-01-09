from collections import OrderedDict

from .DataPacketMemberVars import DataPacketMemberVars
from .TemplateUtility import TemplateUtility


class FortranTemplateUtility(TemplateUtility):
    """
    Internal class for the data packet generator.
    Contains utility functions for generating data packets with fortran
    based task functions.
    """

    FARRAY_MAPPING = {
        "int": "IntVect",
        "real": "RealVect"
    }

    F_HOST_EQUIVALENT = {
        'RealVect': 'real',
        'IntVect': 'int'
    }

    @classmethod
    def iterate_externals(
        cls, connectors: dict,
        size_connectors: dict,
        externals: OrderedDict
    ):
        """
        Iterates the external variables section
        and adds the necessary connectors.

        :param dict connectors: The dictionary containing all connectors
                                for use with CGKit.
        :param dict size_connectors: The dictionary containing all size
                                     connectors for determining the sizes
                                     of each item in the packet.
        :param OrderedDict externals: The dictionary containing the data
                                      for the ptv section in the DataPacket
                                      JSON.
        """
        cls.section_creation(
            cls._EXT, externals, connectors, size_connectors
        )
        connectors[cls._HOST_MEMBERS] = []
        # we need to set nTiles value here as a link,
        # _param does not work as expected.

        nTiles_value = '// Check for overflow first to avoid UB\n' + \
                       '// TODO: Should casting be checked here ' \
                       'or in base class?\n' + \
                       'if (tiles_.size() > INT_MAX)\n\t' \
                       'throw std::overflow_error("' \
                       '[_param:class_name pack]' \
                       ' nTiles was too large for int.");\n' + \
                       '_nTiles_h = static_cast<' \
                       f'{externals["nTiles"]["type"]}>(tiles_.size());'
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
        :param str language: The language to use
        :param int num_arrays: The number of arrays inside tile-in,
                               tile-in-out, tile-out, and tile-scratch.
        """

        cls.section_creation(
            cls._T_MDATA, tilemetadata, connectors, size_connectors
        )
        connectors[cls._T_DESCRIPTOR] = []

        for item, data in tilemetadata.items():
            source = data['source']
            # temporary fix for generating packet for flash-x sim
            if source == 'tile_lbound':
                source = 'tile_loGC'
            elif source == 'tile_ubound':
                source = 'tile_hiGC'
            source = source.replace('tile_', '')

            item_type = data['type']
            size_eq = f"MILHOJA_MDIM * sizeof({item_type})"
            info = DataPacketMemberVars(
                item=item, dtype=item_type, size_eq=size_eq, per_tile=True
            )

            # extend each connector
            connectors[cls._PUB_MEMBERS].extend(
                [f'{item_type}* {info.device};\n']
            )
            connectors[cls._SET_MEMBERS].extend([f'{info.device}{{nullptr}}'])
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {size_eq};\n')
            cls.set_pointer_determination(connectors, cls._T_MDATA, info)

            # data type depends on the language. If there exists a mapping
            # for a fortran data type for the given item_type,
            # use that instead.
            info.dtype = cls.FARRAY_MAPPING.get(item_type, item_type)
            connectors[cls._T_DESCRIPTOR].append(
                f'const auto {source} = tileDesc_h->{source}();\n'
            )

            # ..todo::
            #   * Probably shouldn't always assume I can replace signed
            #   * with unsigned here...
            info.dtype = info.dtype.replace('unsigned', '')

            # indices are 1 based, so bound arrays need to adjust
            # ..todo::
            #       * This is not sufficient for lbound
            fix_index = '+1' if info.dtype == str('IntVect') else ''
            info.dtype = cls.F_HOST_EQUIVALENT[info.dtype]
            construct_host = "[MILHOJA_MDIM] = { " \
                f"{source}.I(){fix_index}, {source}.J(){fix_index}, " + \
                f"{source}.K(){fix_index} }}"

            use_ref = ""
            cls.tile_metadata_memcpy(
                connectors, construct_host, use_ref, info
            )

    @classmethod
    def tile_metadata_memcpy(
        cls, connectors: dict, construct: str,
        use_ref: str, info: DataPacketMemberVars
    ):
        """
        Adds the memcpy portion for the metadata section in a fortran packet.

        :param dict connectors: The dictionary containing all cgkit
                                connectors.
        :param str construct: The generated host variable to copy to the
                              pinned pointer location
        :param str use_ref: Use a reference to the host item.
        :param DataPacketMemberVars info: Contains information for formatting
                                          the name to get variable names.
        """
        connectors[f'memcpy_{cls._T_MDATA}'].extend([
            f'{info.dtype} {info.host}{construct};\n',
            f'char_ptr = static_cast<char*>(static_cast<void*>('
            f'{info.pinned})) + n * {info.size};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>('
            f'{use_ref}{info.host}), {info.size});\n\n',
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

        :param dict connectors: The dict containing all connectors
                                for cgkit.
        :param dict size_connectors: The dict containing all size
                                     connectors for items in the data packet.
        :param OrderedDict tilein: The dict containing the information in
                                   the tile_in section.
        """
        cls.section_creation(cls._T_IN, tilein, connectors, size_connectors)
        for item, data in tilein.items():
            raise NotImplementedError("No test cases for fortran tile_in.")
            # gather all information from tile_in section.
            extents = data['extents']
            mask_in = data['variables_in']
            dtype = data['type']
            index_space = cls.DEFAULT_INDEX_SPACE
            array_size = cls.get_array_size(mask_in, None)

            extents = ' * '.join(f'({item})' for item in extents)
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )
            cls._common_iterate_tile_data(
                data, connectors, info, extents, mask_in, None, cls._T_IN
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

        :param dict connectors: The dict containing all connectors
                                for use with cgkit.
        :param dict size_connectors: The dict containing all size
                                     connectors for items in the JSON.
        :param OrderedDict tileinout: The dict containing the data from
                                      the tile-in-out section of the
                                      datapacket json.
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
            dtype = data['type']
            index_space = cls.DEFAULT_INDEX_SPACE
            array_size = cls.get_array_size(in_mask, out_mask)

            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_data(
                data, connectors, info, extents,
                in_mask, out_mask, cls._T_IN_OUT
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
            * Any pinned pointer that needs to be a datapacket member
              variable should be a private variable.

        :param dict connectors: The dict containing all connectors for use
                                with cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for items in the JSON.
        :param OrderedDict tileout: The dict containing information from the
                                    tile-out section of the data packet JSON.
        """
        cls.section_creation(
            cls._T_OUT, tileout, connectors, size_connectors
        )
        connectors[f'unpack_{cls._T_OUT}'] = []
        for item, data in tileout.items():
            raise NotImplementedError("No test cases for fortran tile_out.")
            # get tile_out information
            out_mask = data['variables_out']
            array_size = cls.get_array_size(None, out_mask)
            index_space = cls.DEFAULT_INDEX_SPACE
            extents = ' * '.join(f'({item})' for item in data[cls._EXTENTS])
            dtype = data['type']
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item,
                dtype=dtype,
                size_eq=f'{extents} * ({unks}) '
                        f'* sizeof({dtype})',
                per_tile=True
            )
            cls._common_iterate_tile_data(
                data, connectors, info, extents, out_mask, cls._T_OUT
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

        :param dict connectors: The dict containing all connectors
                                for use with cgkit.
        :param dict size_connectors: The dict containing all size
                                     connectors for variable sizes.
        :param OrderedDict tilescratch: The dict containing information from
                                        the tilescratch section of the JSON.
        """
        cls.section_creation(
            cls._T_SCRATCH, tilescratch, connectors, size_connectors
        )
        for item, data in tilescratch.items():
            exts = data[cls._EXTENTS]
            dtype = data["type"]
            # need to fill farrays with default args

            extents = ' * '.join(f'({val})' for val in exts)
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * sizeof({dtype})',
                per_tile=True
            )

            cls._common_iterate_tile_scratch(connectors, info)