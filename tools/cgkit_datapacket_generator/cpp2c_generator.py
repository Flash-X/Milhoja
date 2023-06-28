import json_sections as sects
import utility as mutil
import argparse
import json
import os

LICENSE = mutil.C_LICENSE_BLOCK

INSTANTIATE_HYDRO_ADVANCE = """int instantiate_hydro_advance_packet_c({arg_list}, void** packet) {
    if        ( packet == nullptr) {
        std::cerr << "[instantiate_hydro_advance_packet_c] packet is NULL" << std::endl;
        return MILHOJA_ERROR_POINTER_IS_NULL;
    } else if (*packet != nullptr) {
        std::cerr << "[instantiate_hydro_advance_packet_c] *packet not NULL" << std::endl;
        return MILHOJA_ERROR_POINTER_NOT_NULL;
    }

    try {
        *packet = static_cast<void*>(new {name}(dt));
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
    } catch (...) {
        std::cerr << "[instantiate_hydro_advance_packet_c] Unknown error caught" << std::endl;
        return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
    }

    return MILHOJA_SUCCESS;
}"""

def get_arguments_from_json(data, host: list, gpu: list) -> None:
    ext = set()
    keys = [sects.GENERAL, sects.T_MDATA, sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH]
    for item in keys:
        dictionary = data.get(item, {})
        gpu.extend(dictionary)
        for key in dictionary:
            if item in {sects.T_IN, sects.T_IN_OUT, sects.T_OUT, sects.T_SCRATCH} and not isinstance(data[item][key]['extents'], list):
                start = data[item][key]['start' if 'start' in data[item][key] else 'start-in']
                end = data[item][key]['end' if 'end' in data[item][key] else 'end-in']
                # shape = mutil.parse_extents(data[item][key]['extents'], start, end, size="", language=mutil.Language.fortran)
                # ext.update({ f"{ item.replace('(', '').replace(')', '').replace('+1', '').replace('_h', '') }" for item in shape[0].split(' * ')[:-2] })
    host.extend(ext)  

def join_list(start: str, prefix: str, suffix: str, arr: list) -> str:
    return start.join(f'{prefix}{item}{suffix}' for item in arr)

def generate_hydro_advance_bundle_cpp2c(data: dict):
    with open("dr_hydroAdvance_bundle.cxx", 'w') as out:
        out.writelines([
            LICENSE,
            "#include <Milhoja.h>\n",
            "#include <Milhoja_real.h>\n",
            "#include <Milhoja_interface_error_codes.h>\n",
            f'#include "{data["name"]}.h"\n'
            "#ifndef MILHOJA_OPENACC_OFFLOADING\n",
            '#error "This file should only be compiled if using OpenACC offloading"\n', #this will probably get removed.
            '#endif\n',
            'extern "C" {\n',
            '\t// C DECLARATION OF FORTRAN INTERFACE\n',
            '\tvoid dr_hydro_advance_packet_oacc_c2f(\n',
            '\t\tvoid* packet_h\n',
        ])

        extra_streams = data.get(sects.EXTRA_STREAMS, 0)
        out.writelines([
            f'\t\tconst int queue{n}_h\n' for n in range(1, data.get("n-extra-streams", 0)+2)
        ])
        host_args = ["nTiles"]
        gpu_args = ["nTiles"]
        get_arguments_from_json(data, host_args, gpu_args)
        out.writelines([
            join_list(',\n', '\t\tconst int ', '_h', host_args),
            '\n',
            join_list(',\n', '\t\tconst void* ', '_d', gpu_args),
            ')\n'
        ])
        constructor = data.get(sects.GENERAL, {})
        constructor_args = [ f'const milhoja::{constructor[item]}{item}' for item in constructor ]
        out.write(INSTANTIATE_HYDRO_ADVANCE.format(arg_list=','.join(constructor_args), name=data['name']))

def main(data: dict):
    generate_hydro_advance_bundle_cpp2c(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        main(data)