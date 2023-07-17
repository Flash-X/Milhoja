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
import packet_generation_utility as mutil
import os

LICENSE_BLOCK = mutil.F_LICENSE_BLOCK

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
            'packet': {'ftype': None, 'ctype': 'type(C_PTR)'}, 
            **{ f'queue{i}': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)', 'kind': 'acc_handle_kind'} for i in range(1, n_extra_streams+2) }
        }

        arg_order = data[sects.ORDER]
        # nTiles will always be a part of the argument list at the front.
        arg_order.insert(0, 'nTiles')
        gpu_pointers = { 'nTiles': { 'ftype': 'integer', 'ctype': 'type(C_PTR)', 'shape': None} }

        for key,itype in data.get(sects.GENERAL, {}).items():
            ftype = itype.lower()
            if ftype.endswith('int') == 'int': ftype = 'integer'
            gpu_pointers[key] = { 'ftype': ftype, 'ctype': 'type(C_PTR)', 'shape': None }

        for key,name in data.get(sects.T_MDATA, {}).items():
            ftype = mutil.tile_variable_mapping[name].lower()
            if ftype.endswith('int'): ftype='integer'
            shape = [3, 'F_nTiles_h']
            gpu_pointers[name] = {'ftype': ftype, 'ctype': 'type(C_PTR)', 'shape': shape}

        for key,bound in data.get(sects.LBOUND, {}).items():
            ftype = 'integer'
            shape = [str(3 + len(bound)-1), 'F_nTiles_h']
            gpu_pointers[key] = {'ftype': ftype, 'ctype': 'type(C_PTR)', 'shape': shape}

        if 'nTiles' not in extents_set:
            extents_set['nTiles'] = {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)'}

        for section in [sects.T_IN, sects.T_IN_OUT, sects.T_OUT]:
            sect_dict = data.get(section, {})
            for item in sect_dict:
                ftype = sect_dict[item]['type'].lower()
                if ftype.endswith('int'): ftype='integer'
                start = sect_dict[item]['start' if 'start' in sect_dict[item] else 'start-in']
                end = sect_dict[item]['end' if 'end' in sect_dict[item] else 'end-in']
                shape = sect_dict[item]['extents']
                shape.append(f'{end} - {start} + 1')
                shape.append('F_nTiles_h')
                gpu_pointers[item] = {
                    'ftype': ftype,
                    'ctype': 'type(C_PTR)',
                    'shape': shape
                }

        scratch = data.get(sects.T_SCRATCH, {})
        for item in scratch:
            ftype = scratch[item]['type'].lower()
            if ftype.endswith('int'): ftype='integer'
            shape = scratch[item]['extents']
            shape.append('F_nTiles_h')
            gpu_pointers[item] = {
                'ftype': ftype,
                'ctype': 'type(C_PTR)',
                'shape': shape
            }

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

        fp.writelines( [ f'\t{ host_pointers[item]["ctype"] }, intent(IN), value :: C_{item}_h\n' for item in host_pointers if host_pointers[item]['ctype'] ] + ['\n'] )
        fp.writelines( [ f'\t{ gpu_pointers[item]["ctype"] }, intent(IN), value :: C_{item}_d\n' for item in gpu_pointers if gpu_pointers[item]['ctype'] ] + ['\n'] )
        fp.writelines( [ (f"""\t{host_pointers[item]['ftype']}{"" if "kind" not in host_pointers[item] else f"(kind={ host_pointers[item]['kind'] })" } :: F_{item}_h\n""" ) \
                        for item in host_pointers if host_pointers[item]['ftype']] + ['\n'] )

        fp.writelines(
            (
                [
                    f"""\t{gpu_pointers[item]['ftype']}, pointer :: F_{item}_d{'' if not gpu_pointers[item]['shape'] else '(' + ','.join(':' for _ in range(0, len(gpu_pointers[item]["shape"]))) + ')'}\n"""
                    for item in gpu_pointers
                ]
                + ['\n']
            )
        )

        fp.writelines([
            (f"""\tF_{item}_h = INT(C_{item}_h{f', kind={host_pointers[item]["kind"]}' if "kind" in host_pointers[item] else ''})\n""") 
                for item in host_pointers if host_pointers[item]['ftype']
        ] + ['\n'])

        for item in extents_set:
            host_pointers.pop(item)
        fp.writelines([
            f"""\tCALL C_F_POINTER(C_{item}_d, F_{item}_d{f', shape=[{ ", ".join(f"{ext}" for ext in gpu_pointers[item]["shape"])  }]' if gpu_pointers[item]['shape'] else ''})\n"""
                for item in gpu_pointers if gpu_pointers[item]['ftype']
        ] + ['\n'])

        # CALL STATIC FORTRAN LAYER
        fp.writelines([ '\tCALL dr_hydroAdvance_packet_gpu_oacc(',
                        f', &\n'.join( f'\t\tF_{ptr}_h' if host_pointers[ptr]['ftype'] else f'C_{ptr}_h' for ptr in host_pointers ),
                        f', &\n',
                        f', &\n'.join( f'\t\tF_{ptr}_d' for ptr in arg_order )
        ])
        fp.write(')\nend subroutine dr_hydro_advance_packet_oacc_c2f')

def main(data: dict):
    generate_hydro_advance_c2f(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        main(data)
