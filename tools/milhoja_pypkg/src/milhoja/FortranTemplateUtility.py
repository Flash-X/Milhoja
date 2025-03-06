import re

from copy import deepcopy
from collections import OrderedDict

from .DataPacketMemberVars import DataPacketMemberVars
from .parse_helpers import (
    parse_lbound_f, get_initial_index, get_array_size
)
from .TemplateUtility import TemplateUtility
from . import (
    EXTERNAL_ARGUMENT, GRID_DATA_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_INTERIOR_ARGUMENT,
    TILE_ARRAY_BOUNDS_ARGUMENT, TILE_LBOUND_ARGUMENT, LBOUND_ARGUMENT,
    GRID_DATA_LBOUNDS, SCRATCH_ARGUMENT, F2C_TYPE_MAPPING,
    TILE_LEVEL_ARGUMENT, SOURCE_DATATYPES
)


class FortranTemplateUtility(TemplateUtility):
    """
    Internal class for the data packet generator.

    Contains utility functions for generating data packets with fortran
    based task functions.
    """

    def __init__(self, tf_spec):
        super().__init__(tf_spec)

    def iterate_externals(
        self, connectors: dict, size_connectors: dict, externals: OrderedDict,
        dummy_arg_order: list
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
        self.section_creation(
            self._EXT, externals, connectors, size_connectors
        )
        connectors[self._HOST_MEMBERS] = []
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
        connectors[self._NTILES_VALUE] = [nTiles_value]
        self._common_iterate_externals(connectors, externals, dummy_arg_order)

    def iterate_tilemetadata(
        self, connectors: dict, size_connectors: dict,
        tilemetadata: OrderedDict, num_arrays: int
    ):
        """
        Iterates the tilemetadata section of the JSON.

        todo::
            * This code desperately needs to be refactored.
            * lbound arrays should be separated out of this function.

        :param dict connectors: The dict containing all connectors for cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for variable sizes.
        :param OrderedDict tilemetadata: The dict containing information from
                                         the tile-metadata section in the JSON
        :param str language: The language to use
        :param int num_arrays: Unused
        """

        self.section_creation(
            self._T_MDATA, tilemetadata, connectors, size_connectors
        )
        connectors[self._T_DESCRIPTOR] = []
        one_time_mdata = OrderedDict()
        bounds_data = {
            TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_INTERIOR_ARGUMENT,
            TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT, LBOUND_ARGUMENT,
            TILE_ARRAY_BOUNDS_ARGUMENT
        }

        for item, data in tilemetadata.items():
            pure_source = data['source']
            source = pure_source
            short_source = source.replace('tile_', '')
            item_type = data['type']
            interior = TILE_INTERIOR_ARGUMENT
            bound_size_modifier = ""

            if source == interior or source == TILE_ARRAY_BOUNDS_ARGUMENT:
                bound_size_modifier = "2 * "

            # check if the tile source is an array.
            mdim_modifier = "MILHOJA_MDIM"
            if SOURCE_DATATYPES[source] not in ["IntVect", "RealVect"]:
                mdim_modifier = "1"

            size_eq = \
                f"{bound_size_modifier}{mdim_modifier} * sizeof({item_type})"

            lbound = []

            # then it's an lbound array, which means int type and the size of
            # the array is the # of dimensions of the associated array.
            if pure_source == LBOUND_ARGUMENT:
                assoc_array = data["array"]
                item_type = "int"
                array_spec = deepcopy(
                    self.tf_spec.argument_specification(assoc_array)
                )
                source = array_spec["source"]
                if source == EXTERNAL_ARGUMENT:
                    raise NotImplementedError(
                        "Lbound for external arrays not implemented."
                    )
                elif source == GRID_DATA_ARGUMENT:
                    lbound = None
                    struct = array_spec["structure_index"][0].upper()
                    # get starting array value
                    vars_in = array_spec.get('variables_in', None)
                    vars_out = array_spec.get('variables_out', None)
                    init = get_initial_index(vars_in, vars_out)
                    lbound = GRID_DATA_LBOUNDS[struct].format(init)
                    lbound, _ = parse_lbound_f(lbound)
                    lbound = [item.replace("tile_", "") for item in lbound]
                    one_time_mdata[TILE_LBOUND_ARGUMENT] = \
                        {"source": "tile_lbound", "type": "IntVect"}
                    one_time_mdata[TILE_LO_ARGUMENT] = \
                        {"source": "tile_lo", "type": "IntVect"}
                # todo::
                #   * this assumes that any mdata in lbound already exists...
                elif source == SCRATCH_ARGUMENT:
                    lbound = array_spec["lbound"]
                    lbound, _ = parse_lbound_f(lbound)
                    lbound = [item.replace("tile_", "") for item in lbound]
                size_eq = f"{len(lbound)} * sizeof({item_type})"

            info = DataPacketMemberVars(
                item=item, dtype=item_type, size_eq=size_eq, per_tile=True
            )
            # extend each connector
            connectors[self._PUB_MEMBERS].extend(
                [f'{item_type}* {info.device};\n']
            )
            connectors[self._SET_MEMBERS].extend([f'{info.device}{{nullptr}}'])
            connectors[self._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = {size_eq};\n'
            )
            self.set_pointer_determination(connectors, self._T_MDATA, info)

            # data type depends on the language. If there exists a mapping
            # for a fortran data type for the given item_type,
            # use that instead.
            info.dtype = F2C_TYPE_MAPPING.get(item_type, item_type)

            construct_host = ""
            if pure_source != LBOUND_ARGUMENT:
                # adjusting index base
                fix_index = '+1' if pure_source in bounds_data else ''
                info.dtype = F2C_TYPE_MAPPING.get(info.dtype, info.dtype)

                # need to check for tile_interior and tile_arrayBounds args
                # can't assume that lo and hi already exist + each var does
                # not have knowledge of the other
                if source == TILE_INTERIOR_ARGUMENT:
                    construct_host = "[MILHOJA_MDIM * 2] = {" \
                        f"tileDesc_h->lo().I(){fix_index}, tileDesc_h->hi().I(){fix_index}, " \
                        f"tileDesc_h->lo().J(){fix_index}, tileDesc_h->hi().J(){fix_index}, " \
                        f"tileDesc_h->lo().K(){fix_index}, tileDesc_h->hi().K(){fix_index} " \
                        "}"

                elif source == TILE_ARRAY_BOUNDS_ARGUMENT:
                    construct_host = "[MILHOJA_MDIM * 2] = {" \
                        f"tileDesc_h->loGC().I(){fix_index}, tileDesc_h->hiGC().I(){fix_index}, "\
                        f"tileDesc_h->loGC().J(){fix_index}, tileDesc_h->hiGC().J(){fix_index}, "\
                        f"tileDesc_h->loGC().K(){fix_index}, tileDesc_h->hiGC().K(){fix_index} " \
                        "}"

                elif source == TILE_LEVEL_ARGUMENT:
                    one_time_mdata[item] = data
                    construct_host = f"[1] = {{(int){short_source}}}"

                else:
                    one_time_mdata[item] = data
                    construct_host = "[MILHOJA_MDIM] = { " \
                        f"{short_source}.I(){fix_index}, " \
                        f"{short_source}.J(){fix_index}, " \
                        f"{short_source}.K(){fix_index} }}"

            # we can write very specific code if we know that the variable
            # is an lbound argument.
            else:
                # info.dtype = VECTOR_ARRAY_EQUIVALENT[info.dtype]
                # intvect i,j,k start at 0 so we need to add 1 to the
                # index as it is a Fortran.
                # However, anything that's just integers needs to be untouched.
                for idx, value in enumerate(lbound):
                    found = re.search('[a-zA-z]', value)
                    if found:
                        lbound[idx] = f"({value}) + 1"
                construct_host = f"[{len(lbound)}] = {{" +\
                    ','.join(lbound) + "}"

            self.tile_metadata_memcpy(connectors, construct_host, "", info)

        # for tile metadata that is needed by other metadata, but we only
        # want to generate 1 variable for it.
        for item, data in one_time_mdata.items():
            name = item.replace("tile_", "")
            src = data["source"]
            # milhoja uses loGC and hiGC for tile_arrayBounds arrays.
            # so we need this temporary name adjustment.
            if src == TILE_LBOUND_ARGUMENT:
                src = "tile_loGC"
            elif src == TILE_UBOUND_ARGUMENT:
                src = "tile_hiGC"
            src = src.replace("tile_", '')
            connectors[self._T_DESCRIPTOR].append(
                f'const auto {name} = tileDesc_h->{src}();\n'
            )

    def tile_metadata_memcpy(
        self, connectors: dict, construct: str, use_ref: str,
        info: DataPacketMemberVars
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
        connectors[f'memcpy_{self._T_MDATA}'].extend([
            f'{info.dtype} {info.host}{construct};\n',
            f'char_ptr = static_cast<char*>(static_cast<void*>('
            f'{info.pinned})) + n * {info.size};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>('
            f'{use_ref}{info.host}), {info.size});\n\n',
        ])

    def iterate_tile_in(
        self, connectors: dict, size_connectors: dict, tilein: OrderedDict
    ):
        """
        Iterates the tile in section of the JSON.

        todo::
            * Write tests for tile_in data for fortran TFs.

        :param dict connectors: The dict containing all connectors
                                for cgkit.
        :param dict size_connectors: The dict containing all size
                                     connectors for items in the data packet.
        :param OrderedDict tilein: The dict containing the information in
                                   the tile_in section.
        """
        self.section_creation(self._T_IN, tilein, connectors, size_connectors)
        for item, data in tilein.items():
            raise NotImplementedError("No test cases for fortran tile_in.")
            # gather all information from tile_in section.
            extents = data['extents']
            mask_in = data['variables_in']
            dtype = data['type']
            index_space = self.DEFAULT_INDEX_SPACE
            array_size = self.get_array_size(mask_in, None)

            extents = ' * '.join(f'({item})' for item in extents)
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )
            self._common_iterate_tile_data(
                data, connectors, info, extents, mask_in, None, self._T_IN
            )

    def iterate_tile_in_out(
        self, connectors: dict, size_connectors: dict, tileinout: OrderedDict
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
        self.section_creation(
            self._T_IN_OUT, tileinout, connectors, size_connectors
        )
        connectors[f'memcpy_{self._T_IN_OUT}'] = []
        connectors[f'unpack_{self._T_IN_OUT}'] = []
        # unpack all items in tile_in_out
        for item, data in tileinout.items():
            in_mask = data['variables_in']
            out_mask = data['variables_out']
            dtype = data['type']
            index_space = self.DEFAULT_INDEX_SPACE
            array_size = get_array_size(in_mask, out_mask, True)

            extents = ' * '.join(f'({item})' for item in data[self._EXTENTS])
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * ({unks}) * sizeof({dtype})',
                per_tile=True
            )

            self._common_iterate_tile_data(
                data, connectors, info, extents,
                in_mask, out_mask, self._T_IN_OUT
            )

    def iterate_tile_out(
        self,
        connectors: dict,
        size_connectors: dict,
        tileout: OrderedDict
    ):
        """
        Iterates the tileout section of the JSON.

        ..todo::
            * Any pinned pointer that needs to be a datapacket member
              variable should be a private variable.
            * Write tests for fortran tfs that use tile_out data.

        :param dict connectors: The dict containing all connectors for use
                                with cgkit.
        :param dict size_connectors: The dict containing all size connectors
                                     for items in the JSON.
        :param OrderedDict tileout: The dict containing information from the
                                    tile-out section of the data packet JSON.
        """
        self.section_creation(
            self._T_OUT, tileout, connectors, size_connectors
        )
        connectors[f'unpack_{self._T_OUT}'] = []
        for item, data in tileout.items():
            raise NotImplementedError("No test cases for fortran tile_out.")
            # get tile_out information
            out_mask = data['variables_out']
            array_size = self.get_array_size(None, out_mask)
            index_space = self.DEFAULT_INDEX_SPACE
            extents = ' * '.join(f'({item})' for item in data[self._EXTENTS])
            dtype = data['type']
            unks = f'{str(array_size)} + 1 - {str(index_space)}'
            info = DataPacketMemberVars(
                item=item,
                dtype=dtype,
                size_eq=f'{extents} * ({unks}) '
                        f'* sizeof({dtype})',
                per_tile=True
            )
            self._common_iterate_tile_data(
                data, connectors, info, extents, out_mask, self._T_OUT
            )

    def iterate_tile_scratch(
        self, connectors: dict, size_connectors: dict,
        tilescratch: OrderedDict
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
        self.section_creation(
            self._T_SCRATCH, tilescratch, connectors, size_connectors
        )
        for item, data in tilescratch.items():
            exts = data[self._EXTENTS]
            dtype = data["type"]
            # need to fill farrays with default args

            extents = ' * '.join(f'({val})' for val in exts)
            info = DataPacketMemberVars(
                item=item, dtype=dtype,
                size_eq=f'{extents} * sizeof({dtype})',
                per_tile=True
            )

            self._common_iterate_tile_scratch(connectors, info)
