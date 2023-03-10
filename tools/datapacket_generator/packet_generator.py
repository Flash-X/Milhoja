# Author: Wesley Kwiecinski

# Packet generator for Milhoja. Script takes in a 
# JSON object file and generates cpp code based on
# the contents of the file.
# 
# TODO: Create a document that outlines possible options in the json input.
# Right now, the JSON input file has a few options for creating a new packet.
# When adding a variable to any section, you must also specify a milhoja type name. Look in the milhoja repo for possible types
# Possible sections for JSON input:
#   name -              the name of the packet. Every packet must have some sort of identifier.
#   problem -           the problem the created packet is meant to assist with. currently not supported.
#   general -           list of all variables not associated with any tile.
#   tile -              a list of all possible variables include in a tile. also includes suboption array_types.
#       array_types -   all array types to be included in a tile, supported types include FACE[XYZ], CC1/CC2.
#   copy-in -           Any array type you want to add to the copy-in section of the packet.
#   copy-in-out         Any array type you want to add to the copy-in-out section of the packet.
#   copy-out -          Any array type you want to add to the copy-out section of the packet.
#   scratch -           Any array type you want to add to the scratch mem section in the gpu.

import sys
import json
import copy

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

# type constants
SIZE_T = "std::size_t"
# 

# these might not be necessary
vars_and_types = {}
types = set()
array_types = set()
level = 0

# Let's have dictionaries for each section in the json file instead of whatever is going on up there ^
# TODO: Maybe we can create dictionaries derived from sections in the json? Something to think about
general_vars = {}
tile_metadata = {}
cin = {}
cinout = {}
cout = {}
scratch = {}

# It might be beneficial to write helper methods or a wrapper class for files
# to help with consistently writing to the file, so we
# don't have to put {indent} or \n in every line we write.

# header_section_stack = []
# code_section_stack = []
def get_indentation(level):
    return "\t" * level

def generate_cpp_file(parameters):

    def generate_constructor(file, params):
            # function definition
            file.write("%s::%s(void) : milhoja::DataPacket(){}, \n" % (params["name"], params["name"]))
            level = 1
            index = 1
            indent = "\t" * level
            for variable in vars_and_types:
                comma = '' if index == len(vars_and_types) else ','
                file.write(f"{indent}{variable}{{0}}{comma}\n")
                index += 1
            file.write("{\n")

            # constructor code
            file.write(f"{indent}using namespace milhoja;\n")
            for var in ['nxb','nyb','nzb']:
                code.write(f"{indent}unsigned int {var} = 1;\n")
            file.write(f"{indent}Grid::instance().getBlockSize(&nxb, &nyb, &nzb);\n")

            remaining_variables = copy.deepcopy(vars_and_types)

            for type in array_types:
                if type == "CC1" or type == "CC2":
                    var = f"{type}_BLOCK_SIZE"
                    file.writelines([
                        f"{indent}{var} = ",
                        "(nxb + 2 * NGUARD * MILHOJA_K1D) ",
                        "* (nxy + 2 * NGUARD * MILHOJA_K2D) ",
                        "* (nxz + 2 * NGUARD * MILHOJA_K3D);\n"
                    ])
                    remaining_variables.pop(var)
                elif type == "FCX":
                    var = f"{type}_BLOCK_SIZE"
                    file.write(f"{indent}{var} = sizeof(Real) * (nxb+1) * nyb * nzb * NFLUXES;\n")
                    remaining_variables.pop(var)
                elif type == "FCY":
                    var = f"{type}_BLOCK_SIZE"
                    file.write(f"{indent}{var} = sizeof(Real) * nxb * (nyb+1) * nzb * NFLUXES;\n")
                    remaining_variables.pop(var)
                elif type == "FCZ":
                    var = f"{type}_BLOCK_SIZE"
                    file.write(f"{indent}{var} = sizeof(Real) * nxb * nyb * (nzb+1) * NFLUXES;\n")
                    remaining_variables.pop(var)

            # for type in remaining_variables:
            print(remaining_variables)

            file.write("}\n")

    def generate_destructor(file, params):
        packet_name = params["name"]
        file.write(f"~{packet_name}::{packet_name}(void) {{\n")
        file.write(f"}}\n")
        return

    def generate_unpack(file, params):
        packet_name = params["name"]
        level = 1
        indent = level * "\t"
        file.write(f"void {packet_name}::unpack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        file.write(f"}}\n")
        return

    def generate_pack(file, params):
        packet_name = params["name"]
        level = 1
        indent = level * "\t"
        file.write(f"void {packet_name}::pack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        file.write(f"}}\n")
        return
    
    def generate_release_queue(file, params):

        return

    name = parameters["name"]
    with open(parameters["name"] + ".cpp", "w") as code:
        code.write(GENERATED_CODE_MESSAGE)
        code.write(f"#include \"{name}.h\"\n")
        code.write(f"#include <cassert>\n") # do we need certain includes?
        code.write(f"#include <cstring>\n") # for advancing by 1 byte (char *)
        code.write(f"#include <stdexcept>\n")
        code.write(f"#include <Milhoja.h>\n")
        # This is assuming all of the types being passed are milhoja types
        for type in types:
            if type == "Real":  # real header file is named real.h so I kind of have to do this until it gets renamed
                code.write(f"#include <Milhoja_{type.lower()}.h>\n")
            else:
                code.write(f"#include <Milhoja_{type}.h>\n")

        if "problem" in parameters: code.write( "#include \"%s.h\"\n" % parameters["problem"] )
        code.write("#include <Driver.h>\n")

        generate_constructor(code, parameters)   
        generate_destructor(code, parameters) 
        generate_unpack(code, parameters)
        generate_pack(code, parameters)
        generate_release_queue(code, parameters)
    
    return

# Creates a header file based on the given parameters.
# Lots of boilerplate and variable generation.
def generate_header_file(parameters):
    # Every packet json should have a name associated with it.
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
        indent = get_indentation(level)

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

        # pack & unpack methods
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
        header.writelines([
            f"#if MILHOJA_NDIM==3\n",
            f"{indent}milhoja::Stream stream2_;\n",
            f"{indent}milhoja::Stream stream3_;\n",
            "#endif\n"
        ])

        # let's assume everything in the "general" section is some kind of pointer? Not sure.
        if "general" in parameters:
            general = parameters["general"] # general section is the general copy in data information
            for item in general:
                # block_size_var = f"{item}_BLOCK_SIZE"
                vars_and_types[item] = general[item]
                types.add(general[item])
                header.write(f"{indent}milhoja::{general[item]}* {item};\n")    # add a new variable for each item
                # header.write(f"{indent}{SIZE_T}{block_size_var};\n")
                general_vars[item] = general[item]

        # Generate private variables. We need to create a variable
        # for all necessary sizes
        if "tile" in parameters:    # Tile is all of the information associated with a tile
            tile = parameters["tile"]
            if "array_types" in tile:
                for type in tile["array_types"]:
                    # add n elems per type per variable
                    # variable = f"N_ELEMENTS_PER_{type}_PER_VARIABLE"
                    # types.add("{SIZE_T}")
                    # vars_and_types[variable] = "{SIZE_T}"
                    # header.write(f"{indent}{SIZE_T} {variable};\n")
                    array_types.add(type)

                    # # add n elems per variable
                    # variable = f"N_ELEMENTS_PER_{type}"
                    # vars_and_types[variable] = "{SIZE_T}"
                    # header.write(f"{indent}{SIZE_T} {variable};\n")

                    # # add block size variable
                    # variable = f"{type}_BLOCK_SIZE"
                    # vars_and_types[variable] = "{SIZE_T}"
                    # header.write(f"{indent}{SIZE_T} {variable};\n")

                    variable = f"{type}_BLOCK_SIZE"
                    vars_and_types[variable] = f"{SIZE_T}"
                    header.write(f"{indent}{SIZE_T} {variable};\n")

                tile.pop("array_types")

            for item in tile:
                vars_and_types[item] = tile[item]
                types.add(tile[item])
                new_variable = f"{item}_BLOCK_SIZE"
                header.write(f"{indent}{SIZE_T} {new_variable};\n")
                vars_and_types[new_variable] = f"{SIZE_T}"
                
        # Add a size storage var for every type we use
        for type in types:
            if type == f"{SIZE_T}": continue
            var = f"{type}_BLOCK_SIZE"
            header.write(f"{indent}{SIZE_T} {var};\n")

        level -= 1
        indent = get_indentation(level)
        header.write("};\n")

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