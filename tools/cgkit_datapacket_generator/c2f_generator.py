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

@dataclass
class C2FInfo:
    """Scruct containing various attributes of the pointers to be generated for the c2f layer."""
    ftype: str
    ctype: str
    kind: str
    shape: list

def generate_advance_c2f(data):
    """
    Generates the code for passing a data packet from the C layer to the Fortran layer to c2f.F90

    :param dict data: The json file used to generate the data packet associated with this file.
    :return: None
    """
    with open("c2f.F90", 'w') as fp:
        fp.writelines([
            '!! This code was generated using c2f_generator.py.\n',
            mutil.F_LICENSE_BLOCK,
            '#include "Milhoja.h"\n',
            '#ifndef MILHOJA_OPENACC_OFFLOADING\n', 
            '#error "This file should only be compiled if using OpenACC offloading"\n',
            '#endif\n\n',
            'subroutine dr_hydro_advance_packet_oacc_c2f('
        ])

        host_pointers = {
            'packet': C2FInfo('', 'type(C_PTR)', '', []),
            **{ f'queue{i}': C2FInfo('integer', 'integer(MILHOJA_INT)', 'acc_handle_kind', []) for i in range(1, data.get(sects.EXTRA_STREAMS, 0)+2) }
        }

        arg_order = data[sects.ORDER]
        # nTiles will always be a part of the argument list at the front.
        arg_order.insert(0, 'nTiles')
        gpu_pointers = {'nTiles': C2FInfo('integer', 'type(C_PTR)', '', [])}

        for key,itype in data.get(sects.GENERAL, {}).items():
            ftype = itype.lower()
            if 'int' in ftype: ftype = 'integer'
            gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', [])

        for key,name in data.get(sects.T_MDATA, {}).items():
            ftype = mutil.tile_variable_mapping[name].lower()
            if 'int' in ftype: ftype = 'integer'
            gpu_pointers[name] = C2FInfo(ftype, 'type(C_PTR)', '', ['MILHOJA_MDIM', 'F_nTiles_h'])

        # TODO: This needs to change once the bounds section in the data packet json is solidified.
        for key,bound in data.get(sects.LBOUND, {}).items():
            gpu_pointers[key] = C2FInfo('integer', 'type(C_PTR)', '', [str(3 + len(bound)-1), 'F_nTiles_h'])

        extents_set = { 'nTiles': C2FInfo('integer', 'integer(MILHOJA_INT)', '', [])}
        for section in [sects.T_IN, sects.T_IN_OUT, sects.T_OUT]:
            sect_dict = data.get(section, {})
            for item,info in sect_dict.items():
                ftype = info[sects.DTYPE].lower()
                if ftype.endswith('int'): ftype='integer'
                start = info[sects.START if sects.START in data else sects.START_IN]
                end = info[sects.END if sects.END in data else sects.END_IN]
                gpu_pointers[item] = C2FInfo(ftype, 'type(C_PTR)', '', info[sects.EXTENTS] + [f'{end} - {start} + 1', 'F_nTiles_h'])

        scratch = data.get(sects.T_SCRATCH, {})
        for item,info in scratch.items():
            ftype = info[sects.DTYPE].lower()
            if ftype.endswith('int'): ftype='integer'
            gpu_pointers[item] = C2FInfo(ftype, 'type(C_PTR)', '',  info[sects.EXTENTS] + ['F_nTiles_h'])

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

        fp.writelines( [ f'\t{ data.ctype }, intent(IN), value :: C_{item}_h\n' for item,data in host_pointers.items() if data.ctype ] + ['\n'] )
        fp.writelines( [ f'\t{ data.ctype }, intent(IN), value :: C_{item}_d\n' for item,data in gpu_pointers.items() if data.ctype ] + ['\n'] )
        fp.writelines( [ (f"""\t{data.ftype}{"" if not data.kind else f"(kind={ data.kind })" } :: F_{item}_h\n""" ) \
                        for item,data in host_pointers.items() if data.ftype] + ['\n'] )

        fp.writelines(([
                f"""\t{data.ftype}, pointer :: F_{item}_d{'' if not data.shape else '(' + ','.join(':' for _ in range(0, len(data.shape))) + ')'}\n"""
                for item,data in gpu_pointers.items()
            ] + ['\n']
        ))

        fp.writelines(
            [(f"""\tF_{item}_h = INT(C_{item}_h{f', kind={data.kind}' if data.kind else ''})\n""") 
                for item,data in host_pointers.items() if data.ftype] + 
            ['\n']
        )

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

def main(data: dict):
    """Driver function for the c2f generator.
    
    :param dict data: The dictionary containing the data packet JSON.
    :return: None
    """
    generate_advance_c2f(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data[sects.NAME] = os.path.basename(args.JSON).replace('.json', '')
        main(data)
