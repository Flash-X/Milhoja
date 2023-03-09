# Author: Wesley Kwiecinski

# Packet generator for Milhoja. Script takes in a 
# JSON object file and generates cpp code based on
# the contents of the file.

import sys
import json

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

# these might not be necessary
vars_and_types = {}
variables = set()
types = set()

# It might be beneficial to write helper methods
# to help with consistently writing to the file, so we
# don't have to put {indent} or \n in every line we write.

# header_section_stack = []
# code_section_stack = []

def generate_cpp_file(parameters):
    name = parameters["name"]
    with open(parameters["name"] + ".cpp", "w") as code:
        code.write(f"#include \"{name}.h\"\n")
    #     generate_unpack(parameters)
    #     generate_pack(parameters)

    # def generate_unpack(parameters):
    #     return

    # def generate_pack(parameters):
    #     return
    
    return

# Creates a header file based on the given parameters.
# Mostly boilerplate.
def generate_header_file(parameters):
    if not "name" in parameters:
            print("New data packet must include a name.")
            exit(-1)

    with open(parameters["name"] + ".h", "w") as header:

        name = parameters["name"]
        defined = name.upper()
        level = 0
        header.write(GENERATED_CODE_MESSAGE)
        
        # define statements
        header.write(f"#ifndef {defined}\n")
        header.write(f"#define {defined}\n")
        # boilerplate includes
        header.write("#include <Milhoja.h>\n")
        header.write("#include <Milhoja_real.h>\n")
        header.write("#include <Milhoja_DataPacket.h>\n")

        # class definition
        header.write(f"class {name} : public milhoja::DataPacket {{ \n")
        level += 1
        indent = level * "\t"

        # public information
        header.write("public:\n")
        header.write(f"{indent}std::unique_ptr<milhoja::DataPacket> clone(void) const override;\n")
        header.write(indent + f"{name}(void);\n")
        header.write(indent + f"~{name}(void);\n")

        # Constructors & = operations
        header.writelines([f"{indent}{name}({name}&)                  = delete;\n",
                           f"{indent}{name}(const {name}&)            = delete;\n",
                           f"{indent}{name}({name}&& packet)          = delete;\n",
                           f"{indent}{name}& operator=({name}&)       = delete;\n",
                           f"{indent}{name}& operator=(const {name}&) = delete;\n",
                           f"{indent}{name}& operator=({name}&& rhs)  = delete;\n"])

        # copy methods
        header.writelines([
            f"{indent}void pack(void) override;\n",
            f"{indent}void unpack(void) override;\n"
        ])

        # queue methods
        header.writelines([
            f"#if MILHOJA_NDIM == 3 && defined(MILHOJA_OPENACC_OFFLOADING)\n",
            f"{indent}int extraAsynchronousQueue(const unsigned int id) override;\n",
            f"{indent}void releaseExtraQueue(const unsigned int id) override;\n"
            f"#endif\n"
        ])
    
        # private information
        header.write("private:\n")

        # let's assume everything in the "general" section is some kind of pointer? Not sure.
        if "general" in parameters:
            general = parameters["general"]
            for item in general:
                vars_and_types[item] = general[item]
                variables.add(item)
                types.add(general[item])
                header.write(f"{indent}milhoja::{general[item]}* {item};\n")

        # Generate private variables. We need to create a variable
        # for all necessary sizes
        if "tile" in parameters:
            tile = parameters["tile"]
            if "meta" in tile:
                for item in tile["meta"]:
                    vars_and_types[item] = tile["meta"][item]
                    variables.add(item)
                    types.add(tile["meta"][item])
            if "device" in tile:
                for item in tile["device"]:
                    vars_and_types[item] = tile["device"][item]
                    variables.add(item)
                    types.add(tile["device"][item])
                    header.write(f"{indent}std::size_t N_ELEMENTS_PER_{item}_PER_VARIABLE;\n")
                    header.write(f"{indent}std::size_t N_ELEMENTS_PER_{item};\n")
                    header.write(f"{indent}std::size_t {item}_BLOCK_SIZE_BYTES;\n")

        # Add a size storage var for every type we use
        for type in types:
            header.write(f"{indent}std::size_t {type}_SIZE_BYTES;\n")

        level -= 1
        indent = level * "\t"
        header.write("};\n")

        # print(vars_and_types)

        # end start define
        header.write("#endif\n")

    return

def generate_packet_from_json():
    #Check if some file path was passed in
    if len(sys.argv) < 2:
        print("Usage: python packet_generator.py [data_file]")
        return

    with open(sys.argv[1], "r") as file:
        data = json.load(file)
        generate_header_file(data)
        generate_cpp_file(data)

    return

if __name__ == "__main__":
    generate_packet_from_json()