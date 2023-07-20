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
        fp.writelines([
            '!! This code was generated using C2F_generator.py.\n',
            LICENSE_BLOCK,
            '#include "Milhoja.h"\n',
            '#ifndef MILHOJA_OPENACC_OFFLOADING\n', # do we really need to include this?
            '#error "This file should only be compiled if using OpenACC offloading"\n',
            '#endif\n\n',
            'subroutine dr_hydro_advance_packet_oacc_c2f('
        ])

        host_pointers = {
            'packet': {'ftype': None, 'ctype': 'type(C_PTR)'}, 
            **{ f'queue{i}': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)', 'kind': 'acc_handle_kind'} for i in range(1, data.get(sects.EXTRA_STREAMS, 0)+2) }
        }

        arg_order = data[sects.ORDER]
        # nTiles will always be a part of the argument list at the front.
        arg_order.insert(0, 'nTiles')
        gpu_pointers = { 'nTiles': { 'ftype': 'integer', 'ctype': 'type(C_PTR)', 'shape': None} }

        for key,itype in data.get(sects.GENERAL, {}).items():
            print(key, itype)
            ftype = itype.lower()
            if 'int' in ftype: ftype = 'integer'
            gpu_pointers[key] = { 'ftype': ftype, 'ctype': 'type(C_PTR)', 'shape': None }

        for key,name in data.get(sects.T_MDATA, {}).items():
            ftype = mutil.tile_variable_mapping[name].lower()
            if 'int' in ftype: ftype='integer'
            gpu_pointers[name] = { 'ftype': ftype, 'ctype': 'type(C_PTR)', 'shape': [3, 'F_nTiles_h'] }

        # TODO: This needs to change once the bounds section in the data packet json is solidified.
        for key,bound in data.get(sects.LBOUND, {}).items():
            gpu_pointers[key] = { 'ftype': 'integer', 'ctype': 'type(C_PTR)', 'shape': [str(3 + len(bound)-1), 'F_nTiles_h'] }

        extents_set = { 'nTiles': {'ftype': 'integer', 'ctype': 'integer(MILHOJA_INT)'} }
        for section in [sects.T_IN, sects.T_IN_OUT, sects.T_OUT]:
            sect_dict = data.get(section, {})
            for item,data in sect_dict.items():
                ftype = data[sects.DTYPE].lower()
                if ftype.endswith('int'): ftype='integer'
                start = data[sects.START if sects.START in data else sects.START_IN]
                end = data[sects.END if sects.END in data else sects.END_IN]
                gpu_pointers[item] = {
                    'ftype': ftype,
                    'ctype': 'type(C_PTR)',
                    'shape': data[sects.EXTENTS] + [f'{end} - {start} + 1', 'F_nTiles_h']
                }

        scratch = data.get(sects.T_SCRATCH, {})
        for item,data in scratch.items():
            ftype = data[sects.DTYPE].lower()
            if ftype.endswith('int'): ftype='integer'
            gpu_pointers[item] = {
                'ftype': ftype,
                'ctype': 'type(C_PTR)',
                'shape': data[sects.EXTENTS] + ['F_nTiles_h']
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

        fp.writelines( [ f'\t{ data["ctype"] }, intent(IN), value :: C_{item}_h\n' for item,data in host_pointers.items() if data['ctype'] ] + ['\n'] )
        fp.writelines( [ f'\t{ data["ctype"] }, intent(IN), value :: C_{item}_d\n' for item,data in gpu_pointers.items() if data['ctype'] ] + ['\n'] )
        fp.writelines( [ (f"""\t{data['ftype']}{"" if "kind" not in data else f"(kind={ data['kind'] })" } :: F_{item}_h\n""" ) \
                        for item,data in host_pointers.items() if data['ftype']] + ['\n'] )

        fp.writelines(([
                f"""\t{data['ftype']}, pointer :: F_{item}_d{'' if not data['shape'] else '(' + ','.join(':' for _ in range(0, len(data["shape"]))) + ')'}\n"""
                for item,data in gpu_pointers.items()
            ] + ['\n']
        ))

        fp.writelines([
            (f"""\tF_{item}_h = INT(C_{item}_h{f', kind={data["kind"]}' if "kind" in data else ''})\n""") 
                for item,data in host_pointers.items() if data['ftype']
            ] + ['\n']
        )

        for item in extents_set:
            host_pointers.pop(item)
        fp.writelines([
            f"""\tCALL C_F_POINTER(C_{item}_d, F_{item}_d{f', shape=[{ ", ".join(f"{ext}" for ext in data["shape"])  }]' if data['shape'] else ''})\n"""
                for item,data in gpu_pointers.items() if data['ftype']
            ] + ['\n']
        )

        # CALL STATIC FORTRAN LAYER
        fp.writelines([
            '\tCALL dr_hydroAdvance_packet_gpu_oacc(',
            f', &\n'.join( f'\t\tF_{ptr}_h' if data['ftype'] else f'C_{ptr}_h' for ptr,data in host_pointers.items() ),
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
