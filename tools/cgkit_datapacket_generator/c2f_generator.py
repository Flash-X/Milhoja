#!/usr/bin/env python

"""
Generates the C to Fortran layer for data packets.

Note: The same JSON file used to generate the data packets must also be used as a parameter for this script. 

TODO: Bounds are not solidified in the JSON yet. So this code does not deal with bounds in any way.
"""

import os
import argparse
import json
import json_sections as sects
import packet_generation_utility as mutil
from dataclasses import dataclass

_F_LICENSE_BLOCK = """!> @copyright Copyright 2022 UChicago Argonne, LLC and contributors
!!
!! @licenseblock
!! Licensed under the Apache License, Version 2.0 (the "License");
!! you may not use this file except in compliance with the License.
!!
!! Unless required by applicable law or agreed to in writing, software
!! distributed under the License is distributed on an "AS IS" BASIS,
!! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!! See the License for the specific language governing permissions and
!! limitations under the License.
!! @endlicenseblock
!!
!! @file\n
"""

@dataclass
class C2FInfo:
    """Scruct containing various attributes of the pointers to be generated for the c2f layer."""
    ftype: str
    ctype: str
    kind: str
    shape: list

def _generate_advance_c2f(data):
    """
    Generates the code for passing a data packet from the C layer to the Fortran layer to c2f.F90

    :param dict data: The json file used to generate the data packet associated with this file.
    :return: None
    """
    with open("c2f.F90", 'w') as fp:

        data_mapping = {
            'int': 'integer',
            'real': 'real',
            'bool': 'logical'
        }
        
        # # WRITE BOILERPLATE
        fp.writelines([
            '!! This code was generated using c2f_generator.py.\n',
            # _F_LICENSE_BLOCK,
            '#include "Milhoja.h"\n',
            '#ifndef MILHOJA_OPENACC_OFFLOADING\n', 
            '#error "This file should only be compiled if using OpenACC offloading"\n',
            '#endif\n\n',
            'subroutine dr_hydro_advance_packet_oacc_c2f('
        ])

        # initialize host pointers
        host_pointers = {
            'packet': C2FInfo('', 'type(C_PTR)', '', []),
            **{ f'queue{i}': C2FInfo('integer', 'integer(MILHOJA_INT)', 'acc_handle_kind', []) for i in range(1, data.get(sects.EXTRA_STREAMS, 0)+2) }
        }

        # get argument order and insert nTiles (TODO: This affects the CPP2C generator as well. Is it worth it to change this?)
        arg_order = data[sects.ORDER]
        arg_order.insert(0, 'nTiles')
        gpu_pointers = {'nTiles': C2FInfo('integer', 'type(C_PTR)', '', [])} # TODO: this should probably be renamed to device pointers

        # Load all items from general.
        for key,dtype in data.get(sects.GENERAL, {}).items():
            ftype = data_mapping[dtype.lower()]
            gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', [])

        # load all items from tile_metadata.
        for key,name in data.get(sects.T_MDATA, {}).items():
            ftype = mutil.F_HOST_EQUIVALENT[mutil.TILE_VARIABLE_MAPPING[name]].lower()
            ftype = data_mapping[ftype]
            gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', ['MILHOJA_MDIM', 'F_nTiles_h'])

        # TODO: This needs to change once the bounds section in the data packet json is solidified.
        for key,bound in data.get(sects.LBOUND, {}).items():
            gpu_pointers[key] = C2FInfo('integer', 'type(C_PTR)', '', [str(3 + len(bound)-1), 'F_nTiles_h'])

        # need to create an extents set to set the sizes for the fortran arrays.
        extents_set = { 'nTiles': C2FInfo('integer', 'integer(MILHOJA_INT)', '', [])}
        # load all items from array sections, except scratch.
        for section in [sects.T_IN, sects.T_IN_OUT, sects.T_OUT]:
            sect_dict = data.get(section, {})
            for item,info in sect_dict.items():
                ftype = info[sects.DTYPE].lower()
                ftype = data_mapping[ftype]
                
                shape = ['F_nTiles_h']
                if (sects.START in info and sects.END in info) or (sects.START_IN in info and sects.END_IN in info):
                    start = info[sects.START if sects.START in info else sects.START_IN]
                    end = info[sects.END if sects.END in info else sects.END_IN]
                    if start and end:
                        shape.insert(0, f'{end} - {start} + 1')

                gpu_pointers[item] = C2FInfo(ftype, 'type(C_PTR)', '', info[sects.EXTENTS] + shape)

        # finally load scratch data
        scratch = data.get(sects.T_SCRATCH, {})
        for item,info in scratch.items():
            ftype = info[sects.DTYPE].lower()
            if ftype.endswith('int'): ftype='integer'
            gpu_pointers[item] = C2FInfo(ftype, 'type(C_PTR)', '',  info[sects.EXTENTS] + ['F_nTiles_h'])

        host_pointers.update(extents_set)
        # get pointers for every section
        fp.writelines([
            ', &\n'.join(f'C_{item}_h' for item in host_pointers) + ', &\n', # put all host items into func declaration
            ', &\n'.join(f'C_{item}_d' for item in arg_order), # we can assume every item in the TFAL exists in the data packet at this point
            ') bind(c)\n',
            '\tuse iso_c_binding, ONLY : C_PTR, C_F_POINTER\n',
            '\tuse openacc, ONLY : acc_handle_kind\n',
            '\tuse milhoja_types_mod, ONLY : MILHOJA_INT\n',
            '\tuse dr_hydroAdvance_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc\n',
            '\timplicit none\n\n'
        ])

        # write c pointer & host fortran declarations
        fp.writelines( [ f'\t{ data.ctype }, intent(IN), value :: C_{item}_h\n' for item,data in host_pointers.items() if data.ctype ] + ['\n'] )
        fp.writelines( [ f'\t{ data.ctype }, intent(IN), value :: C_{item}_d\n' for item,data in gpu_pointers.items() if data.ctype ] + ['\n'] )
        fp.writelines( [ (f"""\t{data.ftype}{"" if not data.kind else f"(kind={ data.kind })" } :: F_{item}_h\n""" ) \
                        for item,data in host_pointers.items() if data.ftype] + ['\n'] )

        # write Fortran pointer declarations
        fp.writelines(
            [
                f"""\t{data.ftype}, pointer :: F_{item}_d{'' if not data.shape else '(' + ','.join(':' for _ in range(0, len(data.shape))) + ')'}\n"""
                for item,data in gpu_pointers.items()
            ] + 
            ['\n']
        )

        fp.writelines(
            [(f"""\tF_{item}_h = INT(C_{item}_h{f', kind={data.kind}' if data.kind else ''})\n""") 
                for item,data in host_pointers.items() if data.ftype] + 
            ['\n']
        )

        # remove all items in extents set so they don't get passed to the task function
        for item in extents_set:
            host_pointers.pop(item)
        fp.writelines(
            [f"""\tCALL C_F_POINTER(C_{item}_d, F_{item}_d{f', shape=[{ ", ".join(f"{ext}" for ext in data.shape)  }]' if data.shape else ''})\n"""
                for item,data in gpu_pointers.items() if data.ftype] + 
            ['\n']
        )

        # CALL STATIC FORTRAN LAYER
        fp.writelines([
            '\tCALL dr_hydroAdvance_packet_gpu_oacc(',
            f', &\n'.join( f'\t\tF_{ptr}_h' if data.ftype else f'C_{ptr}_h' for ptr,data in host_pointers.items() ),
            f', &\n',
            f', &\n'.join( f'\t\tF_{ptr}_d' for ptr in arg_order )
        ])
        fp.write(')\nend subroutine dr_hydro_advance_packet_oacc_c2f')

def generate_c2f(data: dict):
    """Driver function for the c2f generator.
    
    :param dict data: The dictionary containing the data packet JSON.
    :return: None
    """
    _generate_advance_c2f(data)
    print("Assembled c2f layer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data[sects.NAME] = os.path.basename(args.JSON).replace('.json', '')
        generate_c2f(data)
