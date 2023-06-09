#!/usr/bin/env python

"""
Generates the c2f layer for data packets.

Note: The same JSON file used to generate the data packets must also be used as a parameter
      for this script. 
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
    Generates the code responsible for passing a data packet from the C layer to the Fortran code layer.

    Parameters:
        json - the json file used to generate the data packet associated with this file.
    Returns:
        None
    """
    with open("dr_hydro_advance_packet_oacc_C2F.F90", 'w') as fp:
        extents_set = {}
        fp.writelines([
            '!! This code was generated using C2F_generator.py.\n',
            LICENSE_BLOCK,
            '#include "Milhoja.h"\n',
            '#ifndef MILHOJA_OPENACC_OFFLOADING\n', # do we really need to include this?
            '#error "This file should only be compiled if using OpenACC offloading"\n',
            '#endif\n\n',
            'subroutine dr_hydro_advance_packet_oacc_c2f('
        ])

        n_extra_streams = data.get('n-extra-streams', 0)
        host_pointers = {
            'packet': {'ctype': 'type(C_PTR)'}, 
            **{ f'queue{i}': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)', 'kind': 'acc_handle_kind'} for i in range(1, n_extra_streams+2) }
        }

        arg_order = data[sects.ORDER]
        # nTiles will always be a part of the argument list at the front.
        arg_order.insert(0, 'nTiles')
        gpu_pointers = { 'nTiles': { 'ftype': 'integer', 'ctype': 'type(C_PTR)' } }
        keys = [sects.GENERAL, sects.T_MDATA, sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH]
        for item in keys:
            for key in data.get(item, {}):
                if item == sects.GENERAL:
                    ftype = data[item][key].lower()
                    # ftype = value.lower()
                    if ftype=='int': 'integer'
                    gpu_pointers[f'{key}'] = {
                        'ftype': ftype,
                        'ctype': 'type(C_PTR)'
                    }
                elif item == sects.T_MDATA:
                    ftype = mutil.cpp_equiv[mutil.tile_known_types[key]].lower()
                    if ftype=='int': ftype='integer'
                    shape = [3, 'F_nTiles_h']
                    gpu_pointers[key] = {
                        'ftype': ftype,
                        'ctype': 'type(C_PTR)',
                        'shape': shape
                    }
                elif item in {sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH}:
                    ftype = data[item][key]['type'].lower()
                    if ftype=='int': ftype='integer'
                    start = data[item][key]['start' if 'start' in data[item][key] else 'start-in']
                    end = data[item][key]['end' if 'end' in data[item][key] else 'end-in']
                    item_type = data[item][key]['type']
                    # NOTE: Since this file will never be generated when using CPP, we can always
                    #       use the fortran_size_map.
                    if 'nTiles' not in extents_set:
                        extents_set['nTiles'] = {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)'}
                    shape, nunkvar, indexer, num_elems = mutil.parse_extents(data[item][key]['extents'], start, end, size=item_type, language=mutil.Language.fortran)
                    if isinstance(data[item][key]['extents'], list):
                        shape = data[item][key]['extents']
                    else:
                        ext = [ f"F_{ item.replace('(', '').replace(')', '').replace('+1', '') }" for item in shape.split(' * ')[:-2] ]
                        extents_set = { **extents_set, **{ item.rsplit(' ')[0][2:-2]: {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)'} for item in ext if item not in extents_set } }
                        shape = [ f"F_{ item.replace('(', '').replace(')', '') }" for item in shape.split(' * ')[:-2] ]
                        if start + end != 0: shape.append(nunkvar)
                    shape.append('F_nTiles_h')

                    gpu_pointers[key] = {
                        'ftype': ftype,
                        'ctype': 'type(C_PTR)',
                        'shape': shape
                    } 

        # If the supplied argument order does not contain every argument in each section of the JSON,
        # or if the JSON sections contain extra items not contained in the argument list, then 
        # we abort the generation of this C2F layer for translating the packets.
        if arg_order ^ gpu_pointers.keys():
            print("The argument order list and supplied arguments in the JSON do not match. Exiting")
            fp.close()
            exit(-1)

        host_pointers.update(extents_set)

        # get pointers for every section
        fp.writelines([
            ', &\n'.join(f'C_{item}_h' for item in host_pointers) + ', &\n',
            ', &\n'.join(f'C_{item}_d' for item in arg_order),
            ') bind(c)\n',
            '\tuse iso_c_binding, ONLY : C_PTR, C_F_POINTER\n',
            '\tuse openacc, ONLY : acc_handle_kind\n',
            '\tuse milhoja_types_mod, ONLY : MILHOJA_INT\n',
            '\tuse dr_hydroAdvance_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc\n',
            '\timplicit none\n\n'
        ])

        fp.writelines( [ f'\t{ host_pointers[item]["ctype"] }, intent(IN), value :: C_{item}_h\n' for item in host_pointers ] + ['\n'] )
        fp.writelines( [ f'\t{ gpu_pointers[item]["ctype"] }, intent(IN), value :: C_{item}_d\n' for item in gpu_pointers ] + ['\n'] )
        fp.writelines( [ (f"""\t{host_pointers[item]['ftype']}{"" if "kind" not in host_pointers[item] else f"(kind={ host_pointers[item]['kind'] })" } :: F_{item}_h\n""" ) \
                        for item in host_pointers if 'ftype' in host_pointers[item]] + ['\n'] )

        fp.writelines(
            (
                [
                    f"""\t{gpu_pointers[item]['ftype']}, pointer :: F_{item}_d{'' if 'shape' not in gpu_pointers[item] else '(' + ','.join(':' for _ in range(0, len(gpu_pointers[item]["shape"]))) + ')'}\n"""
                    for item in gpu_pointers
                ]
                + ['\n']
            )
        )

        fp.writelines([
            (f"""\tF_{item}_h = INT(C_{item}_h{f', kind={host_pointers[item]["kind"]}' if "kind" in host_pointers[item] else ''})\n""") 
                for item in host_pointers if 'ftype' in host_pointers[item]
        ] + ['\n'])

        for item in extents_set:
            host_pointers.pop(item)
        fp.writelines([
            f"""\tCALL C_F_POINTER(C_{item}_d, F_{item}_d{f', shape=[{ ", ".join(f"{ext}" for ext in gpu_pointers[item]["shape"])  }]' if 'shape' in gpu_pointers[item] else ''})\n"""
                for item in gpu_pointers if 'ftype' in gpu_pointers[item]
        ] + ['\n'])

        # CALL STATIC FORTRAN LAYER
        fp.writelines([ '\tCALL dr_hydroAdvance_packet_gpu_oacc(',
                        f', &\n'.join( f'\t\tF_{ptr}_h' if 'ftype' in host_pointers[ptr] else f'C_{ptr}_h' for ptr in host_pointers ),
                        f', &\n',
                        f', &\n'.join( f'\t\tF_{ptr}_d' if 'ftype' in gpu_pointers[ptr] else f'C_{ptr}_h' for ptr in arg_order )
        ])
        fp.write(')\nend subroutine dr_hydro_advance_packet_oacc_c2f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        generate_hydro_advance_c2f(data)