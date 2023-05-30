#!/usr/bin/env python

"""

"""

import os

import argparse
import json
import json_sections as sects
import milhoja_utility as mutil
import os

LICENSE_BLOCK = """
!> @copyright Copyright 2022 UChicago Argonne, LLC and contributors
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

def generate_hydro_advance_c2f(data):
    """
    Generates the c to fortran code

    Parameters:
        json - the json file used to generate the data packet.
    Returns:
        None
    """
    with open("dr_hydro_advance_packet_oacc_C2F.F90", 'w') as fp:
        fp.writelines([
            '!! This code was generated using C2F_generator.py.\n',
            LICENSE_BLOCK,
            '#include "Milhoja.h"\n',
            '#ifndef MILHOJA_OPENACC_OFFLOADING\n', # do we really need to include this?
            '#error "This file should only be compiled if using OpenACC offloading"\n',
            '#endif\n\n',
            'subroutine dr_hydro_advance_packet_oacc_c2f('
        ])

        n_extra_streams = 0 if 'n-extra-streams' not in data else data['n-extra-streams'] 
        host_pointers = {
            'C_packet_h': {'ctype': 'type(C_PTR)'}, 
            'C_dataQ_h': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)', 'kind': 'acc_handle_kind'}, 
            **{ f'C_queue{i}_h': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)', 'kind': 'acc_handle_kind'} for i in range(2, n_extra_streams+2) },
            **{ item: {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)'} for item in ['C_nTiles_h', 'C_nxbGC_h', 'C_nybGC_h', 'C_nzbGC_h', 'C_nCcVar_h', 'C_nFluxVar_h']}
        }
        
        gpu_pointers = { 'C_nTiles_d': { 'ftype': 'integer', 'ctype': 'type(C_PTR)' } }
        keys = [sects.GENERAL, sects.T_MDATA, sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH]
        for item in keys:
            if item in data:
                for key in data[item]:
                    if item == sects.GENERAL:
                        ftype = data[item][key].lower()
                        if ftype=='int': 'integer'
                        gpu_pointers[f'C_{key}_start_d'] = {
                            'ftype': ftype,
                            'ctype': 'type(C_PTR)'
                        }
                    elif item == sects.T_MDATA:
                        ftype = mutil.cpp_equiv[mutil.tile_known_types[key]].lower()
                        if ftype=='int': ftype='integer'
                        shape = [3, 'F_nTiles_h']
                        gpu_pointers[f'C_{key}_start_d'] = {
                            'ftype': ftype,
                            'ctype': 'type(C_PTR)',
                            'shape': shape
                        }
                    elif item in {sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH}:
                        ftype = data[item][key]['type'].lower()
                        if ftype=='int': ftype='integer'
                        start = data[item][key]['start' if 'start' in data[item][key] else 'start-in']
                        end = data[item][key]['end' if 'end' in data[item][key] else 'end-in']
                        # NOTE: Since this file will never be generated when using CPP, we can always
                        #       use the fortran_size_map.
                        shape, nunkvar, indexer = mutil.parse_extents(data[item][key]['extents'], start, end, '')
                        shape = mutil.fortran_size_map[indexer].format(unk=nunkvar, size='')
                        shape = [ f"F_{ item.replace('(', '').replace(')', '') }" for item in shape.split(' * ')[:-2] ]
                        if start + end != 0: shape.append(nunkvar)
                        shape.append('F_nTiles_h')
                        
                        gpu_pointers[f'C_{key}_start_d'] = {
                            'ftype': ftype,
                            'ctype': 'type(C_PTR)',
                            'shape': shape
                        } 

        bundle = {**host_pointers, **gpu_pointers}
        # get pointers for every section
        fp.write( ', &\n'.join(f'{item}' for item in bundle ) )
        # fp.writelines( [ f'{item}, &\n' for item in {**host_pointers, **gpu_pointers} ] )
        fp.write('\n')
        fp.writelines([
            ') bind(c)\n',
            '\tuse iso_c_binding, ONLY : C_PTR, & C_F_POINTER\n',
            '\tuse openacc, ONLY : acc_handle_kind\n',
            '\tuse milhoja_types_mod, ONLY : MILHOJA_INT\n',
            '\tuse dr_hydroAdvance_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc\n',
            '\timplicit none\n\n'
        ])

        fp.writelines( [ f'\t{ bundle[item]["ctype"] }, intent(IN), value :: {item}\n' for item in bundle ] + ['\n'] )
        fp.writelines( [ (f"""\t{host_pointers[item]['ftype']}{"" if "kind" not in host_pointers[item] else f"(kind={ host_pointers[item]['kind'] })" } :: {'F_' + item[ len("C_"): ]}\n""" ) \
                        for item in host_pointers if 'ftype' in host_pointers[item]] + ['\n'] )

        fp.writelines([
            (f"""\t{gpu_pointers[item]['ftype']}, pointer :: {'F_' + item[ len("C_"): ]}{'' if 'shape' not in gpu_pointers[item] else '(' + ','.join(':' for idx in range(0, len(gpu_pointers[item]["shape"]))) + ')' }\n""")
                for item in gpu_pointers
        ] + ['\n'])

        fp.writelines([
            (f"""\t{'F_' + item[ len("C_"): ]} = INT({item}{f', kind={host_pointers[item]["kind"]}' if "kind" in host_pointers[item] else ''})\n""") 
                for item in host_pointers if 'ftype' in host_pointers[item]
        ] + ['\n'])

        for item in ['C_nTiles_h', 'C_nxbGC_h', 'C_nybGC_h', 'C_nzbGC_h', 'C_nCcVar_h', 'C_nFluxVar_h']:
            host_pointers.pop(item)
        bundle = {**host_pointers, **gpu_pointers}
        fp.writelines([
            f"""\tCALL C_F_POINTER({item}, F_{item[ len('C_'): ]}{f', shape=[{ ", ".join(f"{ext}" for ext in gpu_pointers[item]["shape"])  }]' if 'shape' in gpu_pointers[item] else ''})\n"""
                for item in gpu_pointers if 'ftype' in gpu_pointers[item]
        ] + ['\n'])

        # CALL STATIC FORTRAN LAYER
        fp.write('\tCALL dr_hydroAdvance_packet_gpu_oacc(')
        fp.write( f', &\n'.join( f'\t\tF_' + ptr[ len('C_'): ] if 'ftype' in bundle[ptr] else ptr for ptr in bundle ) )
        fp.write(')\n')
        fp.write('end subroutine dr_hydro_advance_packet_oacc_c2f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        generate_hydro_advance_c2f(data)