#!/usr/bin/env python

"""
Generates the C to Fortran layer for data packets.

Note: The same JSON file used to generate the data packets
      must also be used as a parameter for this script. 

      ..todo:: 
        * Add new tile_metadata functionality (cell_volumes, cell coords, etc.)
        * Add functionality for external arguments with extents.
"""

from TemplateUtility import TemplateUtility
from FortranTemplateUtility import FortranTemplateUtility
from pathlib import Path
from dataclasses import dataclass
from milhoja import AbcCodeGenerator
from milhoja import BasicLogger

@dataclass
class C2FInfo:
    """
    Scruct containing various attributes of the 
    pointers to be generated for the c2f layer.
    """
    ftype: str
    ctype: str
    kind: str
    shape: list

class C2FortranLayerGenerator(AbcCodeGenerator):

    def __init__(
        self,
        tf_spec,
        c2f_file,
        indent,
        log_level,
        n_ex_streams,
        externals,
        tile_metadata,
        tile_in,
        tile_in_out,
        tile_out,
        tile_scratch
    ):
        self._c2f_file = c2f_file
        self._n_extra_streams = n_ex_streams
        self._externals = externals
        self._tile_metadata = tile_metadata
        self._tile_in = tile_in
        self._tile_in_out = tile_in_out
        self._tile_out = tile_out
        self._scratch = tile_scratch

        logger = BasicLogger(log_level)
        super().__init__(tf_spec, "", str(c2f_file), indent, self.__class__.__name__, logger)

    def generate_header_code(self, overwrite):
        """No implementation for generating header code for c2f layer."""
        return

    def generate_source_code(self, overwrite):
        self._generate_advance_c2f(overwrite)

    def _generate_advance_c2f(
        self, 
        overwrite
    ):
        """
        Generates the code for passing a data packet from 
        the C layer to the Fortran layer to c2f.F90

        :param dict data: The json file used to generate the data packet associated with this file.
        """
        if self._c2f_file.is_file():
            self._warn("{self.generator.c2f_file} already exists.")
            if not overwrite:
                self._error("Abort")

        with open(self._c2f_file, 'w') as fp:
            # should size_t be translated if using fortran?
            data_mapping = {
                'int': 'integer',
                'real': 'real',
                'bool': 'logical'
            }
            
            # # WRITE BOILERPLATE
            fp.writelines([
                '#include "Milhoja.h"\n',
                '#ifndef MILHOJA_OPENACC_OFFLOADING\n', 
                '#error "This file should only be compiled if using OpenACC offloading"\n',
                '#endif\n\n',
                f'subroutine {self._tf_spec.name}_C2F('
            ])

            # initialize host pointers
            host_pointers = {
                'packet': C2FInfo('', 'type(C_PTR)', '', []),
                **{ f'queue{i}': C2FInfo('integer', 'integer(MILHOJA_INT)', 'acc_handle_kind', [])
                    for i in range(1, self._n_extra_streams+2)}
            }

            # get argument order and insert nTiles
            # (This affects the CPP2C generator as well.)
            arg_order = ["nTiles"] + self._tf_spec.dummy_arguments
            # TODO: this should probably be renamed to device pointers
            gpu_pointers = {'nTiles': C2FInfo('integer', 'type(C_PTR)', '', [])}

            # Load all items from general.
            for key,data in self._externals.items():
                ftype = data_mapping[data["type"].lower()]
                gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', [])

            # load all items from tile_metadata.
            for key,data in self._tile_metadata.items():
                ftype = FortranTemplateUtility.F_HOST_EQUIVALENT[TemplateUtility.TILE_VARIABLE_MAPPING[data["source"]]].lower()
                ftype = data_mapping[ftype]
                gpu_pointers[key] = C2FInfo(
                    ftype, 'type(C_PTR)', '', ['MILHOJA_MDIM', 'F_nTiles_h']
                )

            # need to create an extents set to set the sizes for the fortran arrays.
            extents_set = {'nTiles': C2FInfo('integer', 'integer(MILHOJA_INT)',  '', [])}
            # load all items from array sections, except scratch.
            for section in [self._tile_in, self._tile_in_out, self._tile_out]:
                for item,data in section.items():
                    ftype = data["type"].lower()
                    ftype = data_mapping[ftype]
                    
                    shape = ['F_nTiles_h']
                    var_in = data.get("variables_in", None)
                    var_out = data.get("variables_out", None)

                    if var_in or var_out:
                        mask = var_in if var_in else var_out
                        shape.insert(0, f'{mask[1]} - {mask[0]} + 1')

                    gpu_pointers[item] = C2FInfo(
                        ftype, 'type(C_PTR)', '', data["extents"] + shape
                    )

            # finally load scratch data
            for item,data in self._scratch.items():
                ftype = data['type'].lower()
                if ftype.endswith('int'): ftype='integer'
                gpu_pointers[item] = C2FInfo(
                    ftype, 'type(C_PTR)', '',  data["extents"] + ['F_nTiles_h']
                )

            host_pointers.update(extents_set)
            # get pointers for every section
            fp.writelines([
                # put all host items into func declaration
                ', &\n'.join(f'C_{item}_h' for item in host_pointers) + ', &\n', 
                # we can assume every item in the TFAL exists in the data packet at this point
                ', &\n'.join(f'C_{item}_d' for item in arg_order), 
                ') bind(c)\n',
                '\tuse iso_c_binding, ONLY : C_PTR, C_F_POINTER\n',
                '\tuse openacc, ONLY : acc_handle_kind\n',
                '\tuse milhoja_types_mod, ONLY : MILHOJA_INT\n',
                f'\tuse {self._tf_spec.name}_bundle_mod, ONLY : {self._tf_spec.name}\n',
                '\timplicit none\n\n'
            ])

            # write c pointer & host fortran declarations
            fp.writelines([
                f'\t{data.ctype}, intent(IN), value :: C_{item}_h\n'
                for item,data in host_pointers.items() if data.ctype] +
                ['\n']
            )
            fp.writelines([
                f'\t{data.ctype}, intent(IN), value :: C_{item}_d\n' 
                for item,data in gpu_pointers.items() if data.ctype ] +
                ['\n']
            )
            fp.writelines([
                (f"""\t{data.ftype}{"" if not data.kind else f"(kind={ data.kind })" } :: F_{item}_h\n""")
                for item,data in host_pointers.items() if data.ftype] +
                ['\n']
            )

            # write Fortran pointer declarations
            fp.writelines([
                f"""\t{data.ftype}, pointer :: F_{item}_d{'' if not data.shape else '(' + ','.join(
                    ':' for _ in range(0, len(data.shape))) + ')'}\n"""
                for item,data in gpu_pointers.items()] + 
                ['\n']
            )

            fp.writelines([
                (f"\tF_{item}_h = INT(C_{item}_h{f', kind={data.kind}' if data.kind else ''})\n")
                for item,data in host_pointers.items() if data.ftype] + 
                ['\n']
            )

            # remove all items in extents set so they don't get passed to the task function
            for item in extents_set:
                host_pointers.pop(item)

            c2f_pointers = [
                f"""\tCALL C_F_POINTER(C_{item}_d, F_{item}_d{
                    f', shape=[{", ".join(ext for ext in data.shape)}]' 
                    if data.shape else ''
                })\n"""
                for item,data in gpu_pointers.items() if data.ftype
            ]
            fp.writelines(c2f_pointers + ['\n'])

            # CALL STATIC FORTRAN LAYER
            fp.writelines([
                f'\tCALL {self._tf_spec.name}(',
                ', &\n'.join(f'\t\tF_{ptr}_h' if data.ftype else f'C_{ptr}_h' 
                            for ptr,data in host_pointers.items()),
                ', &\n',
                ', &\n'.join(f'\t\tF_{ptr}_d' for ptr in arg_order)
            ])
            fp.write(f')\nend subroutine {self._tf_spec.name}_C2F')