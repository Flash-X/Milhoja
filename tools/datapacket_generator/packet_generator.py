# Author: Wesley Kwiecinski

# Packet generator for Milhoja. Script takes in a 
# JSON file and generates cpp code for a data packet
# based on the contents of the json file.
# 
# TODO: Create a document that outlines possible options in the json input.
# Right now, the JSON input file has a few options for creating a new packet.
# When adding a variable to any section, you must also specify a milhoja type name. Look in the milhoja repo for possible types
# Possible sections for JSON input:
#   name -              Packet identifier. Must be included.
#   general -           list of all variables not associated with any tile.
#   tile -              Includes suboption array_types and metadata.
#       metadata -      Per tile variables.
#       array_types -   all array types to be included in a tile, supported types include FACE[XYZ], CC1/CC2.
#   copy-in -           Any array type you want to add to the copy-in section of the packet.
#   copy-in-out         Any array type you want to add to the copy-in-out section of the packet.
#   copy-out -          Any array type you want to add to the copy-out section of the packet.
#   scratch -           Any array type you want to add to the scratch mem section in the gpu.

import sys
import json
import copy

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

# All possible sections
GENERAL = "general"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_OUT = "tile-out"
# 

# type constants
SIZE_T = "std::size_t"
BLOCK_SIZE = "_BLOCK_SIZE_HELPER"
N_TILES = "nTiles"
SCRATCH = "_scratch_d"
TILE_DESC = "tileDesc_h"
# 

# these might not be necessary
vars_and_types = {}
level = 0

# Let's have dictionaries for each section in the json file instead of whatever is going on up there ^
# TODO: Maybe we can create dictionaries derived from sections in the json? Something to think about
known_sections = set()

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
            file.write("%s::%s(Real* dt = nullptr) : milhoja::DataPacket(){}, \n" % (params["name"], params["name"]))
            level = 1
            index = 1
            indent = "\t" * level
            # we probably don't need to initialize all of the vars since we're generating everything
            for variable in vars_and_types:
                comma = '' if index == len(vars_and_types) else ','
                file.write(f"{indent}{variable}{{0}}{comma}\n")
                index += 1
            file.write("{\n")

            # some misc constructor code for calculating block sizes.
            file.write(f"{indent}using namespace milhoja;\n")
            for var in ['nxb','nyb','nzb']:
                code.write(f"{indent}unsigned int {var} = 1;\n")
            file.write(f"{indent}Grid::instance().getBlockSize(&nxb, &nyb, &nzb);\n")

            # TODO: I think we need to be careful here. Do we automatically assume that 'extents' and 'type' exist
            # when given dictionary instead of a scalar? Do we want to force the user to include those 2 keys when 
            # specifying an array type?
            for section in known_sections:
                for item in params[section]:
                    if not isinstance(params[section][item], (dict, list)):
                        file.write(f"{indent}{item}{BLOCK_SIZE} = sizeof({params[section][item]});\n")
                    else:
                        extents = params[section][item]['extents']
                        type = params[section][item]['type']
                        file.write(f"{indent}{item}{BLOCK_SIZE} = {' * '.join(extents)} * sizeof({type});\n")
            
            # We might need to stick all potential constants in the constructor header.\
            # We should not do this 
            if GENERAL in params:
                if 'dt' in params[GENERAL]:
                    file.write(f"{indent}this.dt = dt;\n")

            file.write("}\n\n")

    # TODO: Do we need anything in this destructor?
    def generate_destructor(file, params):
        packet_name = params["name"]
        file.write(f"~{packet_name}::{packet_name}(void) {{\n")
        file.write(f"}}\n\n")
        return

    # TODO: Generate unpack method
    def generate_unpack(file, params):
        packet_name = params["name"]
        level = 1
        indent = level * "\t"
        file.write(f"void {packet_name}::unpack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        file.write(f"{indent}if (tiles_.size() <= 0) {{\n")
        file.write(f"{indent}}}\n")
        file.write(f"}}\n\n")
        return

    # TODO: How can we make pack method generation easier?
    # TODO: {N_TILES} are a guaranteed part of general, same with PacketContents. We also want to consider letting the user add numeric constants to general
    def generate_pack(file, params):
        packet_name = params["name"]
        func_name = "pack"
        level = 1
        indent = level * "\t"
        tab = "\t"
        file.write(f"void {packet_name}::pack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        
        file.writelines([
            f"{indent}std::string errMsg = isNull();\n",
            f"{indent}if (errMsg != \"\") {{\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent} else if (tiles_.size() == 0) {{\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent}}}\n"
            f"{indent}Grid& grid = Grid::instance();\n"
        ])

        # cc1_block_size_exists = False
        # cc2_block_size_exists = False
        # # generate scratch section
        file.write(f"{indent}// Scratch section\n")

        # Note CC2 (aka tile-out) also doubles as a scratch section.
        nScratchArrs = 0
        file.write(f"{indent}{SIZE_T} nScratchPerTileBytes = 0")
        for item in [T_SCRATCH, T_OUT]:
            for var in params.get(item, []):
                nScratchArrs += 1
                file.write(f" + {var}{BLOCK_SIZE}")
        file.write(f";\n{indent}unsigned int nScratchArrays = {nScratchArrs};\n")

        # Non tile specific data
        file.write(f"\n{indent}// non tile specific data\n")
        file.write(f"{indent}{SIZE_T} {N_TILES} = tiles_.size();\n")
        file.write(f"{indent}{SIZE_T} nCopyInBytes = sizeof({N_TILES}) ")
        for item in params.get(GENERAL, []):
            file.write(f"+ {item}{BLOCK_SIZE} ")
        file.write(f"+ {N_TILES} * sizeof(PacketContents);\n")

        # Tile specific data.
        # TODO: Can we specify FArray4D in the JSON file?
        #       Maybe convert cpp variable names to macros or constants.
        #       
        file.write(f"{indent}{SIZE_T} nBlockMetadataPerTileBytes = (nScratchArrays + 1) * sizeof(FArray4D)")
        for item in params.get(T_MDATA, []):
            file.write(f" + {item}{BLOCK_SIZE}")
        file.write(f";\n")

        # Remember tile-in-out is just considered tile-out.
        # Also remember that we are pointing to the next free available byte in memory.
        file.write(f"{indent}{SIZE_T} nCopyInOutDataPerTileBytes = 0")
        if T_IN in params:
            for item in params[T_IN]:
                file.write(f" + {item}{BLOCK_SIZE}")
        file.write(f";\n")

        # # TODO: copy in block data?

        # # generate cout
        file.write(f"{indent}// Copy out section\n")
        file.writelines([
            f"{indent}nCopyToGpuBytes_ = nCopyInBytes + ({N_TILES} * nBlockMetaDataPerTileBytes) + ({N_TILES} * nCopyInOutDataPerTileBytes);\n",
            f"{indent}nReturnToHostBytes_ = {N_TILES} * nCopyInOutDataPerTileBytes;\n",
            f"{indent}{SIZE_T} nBytesPerPacket = {N_TILES} * nScratchPerTileBytes + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes) + ({N_TILES} * nCopyInOutDataPerTileBytes);\n\n"
        ])

        file.writelines([
            f"{indent}stream_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}if (!stream_.isValid()) {{\n",
            f"{indent * 2}throw std::runtime_error(\"[{packet_name}::pack] Unable to acquire stream\");\n{indent}}}\n"
            f"# if MILHOJA_NDIM == 3\n",
            f"{indent}stream2_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}stream3_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}if (!stream2_.isValid() || !stream3_.isValid()) {{\n",
            f"{indent * 2}throw std::runtime_error(\" [{packet_name}::pack] Unable to acquire extra streams\");\n{indent}}}\n#endif\n"
        ])

        # # acquire gpu mem
        file.write(f"{indent}RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - {N_TILES} * nScratchPerTileBytes, &packet_p_, nBytesPerPacket, &packet_d_);\n")
        file.writelines([
            f"{indent}location_ = PacketDataLocation::CC1;\n" # TODO: We need to change this
            f"{indent}char* scratchStart_d = static_cast<char*>(packet_d_);\n",
            f"{indent}copyInStart_p_ = static_cast<char*>(packet_p_);\n",
            f"{indent}copyInStart_d_ = scratchStart_d + {N_TILES} nScratchPerTileBytes;\n",
            f"{indent}copyInOutStart_p_ = copyInStart_p_ + nCopyInBytes + {N_TILES} * nBlockMetadataPerTileBytes;\n",
            f"{indent}copyInOutStart_d_ = copyInStart_d_ + nCopyInBytes + {N_TILES} * nBlockMetadataPerTileBytes;\n",
            f"{indent}if (pinnedPtrs_) {{\n",
            f"{indent * 2}throw std::logic_error(\"{packet_name}::pack Pinned pointers already exist\");\n",
            f"{indent}}}\n",
            f"{indent}pinnedPtrs_ = new BlockPointersPinned[{N_TILES}];\n"
        ])

        # scratch section? Again?

        # copy-in section
        file.writelines([
            f"{indent}static_assert(sizeof(char) == 1, \"Invalid char size\")\n", # we might not need this anymore
            f"{indent}char* ptr_p = copyInStart_p_;\n",
            f"{indent}char* ptr_d = copyInStart_d_;\n",
        ])

        # automatically generate nTiles data PacketContents
        file.writelines([
            f"{indent}std::memcpy((void*)ptr_p, (void*)&{N_TILES}, sizeof({SIZE_T}));\n",
            f"{indent}ptr_p += sizeof({SIZE_T});\n",
            f"{indent}ptr_d += sizeof({SIZE_T});\n",
            f"{indent}contents_p_ = static_cast<PacketContents*>((void*)ptr_p);\n",
            f"{indent}contents_d_ = static_cast<PacketContents*>((void*)ptr_d);\n",
            f"{indent}ptr_p += {N_TILES} * sizeof(PacketContents);\n",
            f"{indent}ptr_d += {N_TILES} * sizeof(PacketContents);\n"
        ])

        for item in params.get(GENERAL, []):
            file.writelines([
                # f"{item} = static_cast<{params['general'][item]}*>((void*)ptr_d)"
                f"{indent}std::memcpy((void*)ptr_p, (void*){item}, {item}{BLOCK_SIZE});\n",
                f"{indent}ptr_p += sizeof({item}{BLOCK_SIZE});\n",
                f"{indent}ptr_d += sizeof({item}{BLOCK_SIZE});\n"
            ])

        file.writelines([
            f"{indent}char* CC_data_p = copyInOutStart_p_;\n",
            f"{indent}char* CC_data_d = copyInOutStart_d_;\n",
            f"{indent}char* CC{SCRATCH} = scratchStart_d;\n",
        ])

        cc_dict = {}
        
        # Note tile-out is also considered scratch data.
        scr = sorted(list(params.get(T_SCRATCH, {}).keys()) + list(params.get(T_OUT, {}).keys()))
        for i in range(1, len(scr)):
            if i == 1:
                file.write(f"{indent}char* {scr[i]}{SCRATCH} = CC{SCRATCH} + {scr[i-1]}{BLOCK_SIZE};\n")
            else:
                file.write(f"{indent}char* {scr[i]}{SCRATCH} = {scr[i-1]}{SCRATCH} + {scr[i-1]}{BLOCK_SIZE};\n")
        file.write(f"{indent}PacketContents* tilePtrs_p = contents_p_;\n")
            
        # tile specific metadata.
        file.write(f"{indent}for ({SIZE_T} n=0; i < {N_TILES}; ++n, ++tilePtrs_p) {{\n")
        indent = "\t" * 2
        file.writelines([
            f"{indent}Tile* tileDesc_h = tiles_[n].get();\n",
            f"{indent}if (tileDesc_h == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Bad tileDesc.\");\n",
            f"{indent}const RealVect deltas = {TILE_DESC}->deltas();\n"
            f"{indent}const IntVect lo = {TILE_DESC}->lo();\n"
            f"{indent}const IntVect hi = {TILE_DESC}->hi();\n"
            f"{indent}const IntVect loGC = {TILE_DESC}->loGC();\n"
            f"{indent}const IntVect hiGC = {TILE_DESC}->hiGC();\n"
            f"{indent}Real* data_h = {TILE_DESC}->dataPtr();\n"
            f"{indent}if (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid ptr to data in host memory.\");\n"
        ])

        for item in params.get(T_IN, []):
            file.writelines([
                f"{indent}std::memcpy((void*)CC_data_p, (void*)data_h, {item}{BLOCK_SIZE});\n",
                f"{indent}pinnedPtrs_[n].CC1_data = static_cast<{params[T_IN][item]['type']}*>((void*)CC_data_p);\n",
                f"{indent}pinnedPtrs_[n].CC2_data = nullptr;\n\n"
            ])

        for item in params.get(T_MDATA, []):
            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<{params[T_MDATA][item]}*>((void*)ptr_d);\n",
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"{indent}ptr_p += {item}{BLOCK_SIZE};\n"
                f"{indent}ptr_d += {item}{BLOCK_SIZE};\n\n"
            ])

        # CCs?
        type = 'FArray4D'
        ccs = list(params.get(T_IN)) + list(params.get(T_OUT, []))
        ccs = {**params.get(T_IN, {}), **params.get(T_OUT, {})}
        for item in ccs:
            cc_var = "CC_data_d" if item == "CC1" else f"CC{SCRATCH}"
            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<FArray4D*>((void*)ptr_d);\n",
                f"{indent}{type} CC_d{{static_cast<{ccs[item]['type']}*>((void*){cc_var}), loGC, hiGC, {ccs[item]['extents'][-1]}}};\n", # end of extents array is nunkvars.
                f"{indent}std::memcpy((void*)ptr_p, (void*)&CC_d, sizeof(FArray4D));\n",
                f"{indent}ptr_p += sizeof(FArray4D);\n",
                f"{indent}ptr_d += sizeof(FArray4D);\n\n"
            ])
            type = ''

        for item in params.get(T_IN):
            file.writelines([
                f"{indent}CC_data_p += {item}{BLOCK_SIZE};\n",
                f"{indent}CC_data_d += {item}{BLOCK_SIZE};\n\n"
            ])

        for item in params.get(T_OUT):
            file.write(f"{indent}CC{SCRATCH} += nScratchPerTileBytes;\n")

        # Generate FACE[XYZ]
        # TODO: Add array dimensionality specification?
        possible_xyz = set(['FCX', 'FCY', 'FCZ'])
        type = "IntVect"
        for item in sorted(list(params.get(T_SCRATCH, []))):
            list_ndim = []
            if item in possible_xyz: possible_xyz.remove(item)
            # I don't really like doing this
            if item == "FCX": list_ndim = ['hi.I()+1', 'hi.J()', 'hi.K()']
            elif item == "FCY": list_ndim = ['hi.I()', 'hi.J()+1', 'hi.K()']
            elif item == "FCX": list_ndim = ['hi.I()', 'hi.J()', 'hi.K()+1']

            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<FArray4D*>((void*)ptr_d);\n"
                f"{indent}{type}fHi = IntVect{{LIST_NDIM({ ', '.join(list_ndim) })}};\n"
                f"{indent}FArray4D {item}_d{{static_cast<{params[T_SCRATCH][item]['type']}*>((void*) {item}{SCRATCH}), lo, fHi, NFLUXES)}};\n"
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}_d, sizeof(FArray4D));\n",
                f"{indent}ptr_p += sizeof(FArray4D);\n"
                f"{indent}ptr_d += sizeof(FArray4D);\n"
                f"{indent}{item}{SCRATCH} += nScratchPerTileBytes;\n\n"
            ])
            type = ""
        # Not sure about this....
        for item in possible_xyz:
            file.write(f"{indent}tilePtrs_p->{item}_d = nullptr;\n")

        indent = "\t"

        file.write(f"{indent}}}\n")
        
        file.write(f"}}\n\n")
        return
    
    def generate_clone(file, params):
        packet_name = params["name"]
        file.writelines([
            f"std::unique_ptr<milhoja::DataPacket> {packet_name}::clone(void) const {{\n",
            f"\treturn std::unique_ptr<milhoja::DataPacket>{{ new {packet_name}{{}} }};\n"
            f"}}"
        ])

    def generate_release_queues(file, params):
        packet_name = params['name']
        # extra async queues
        func_name = "extraAsynchronousQueue"
        file.writelines([
            f"int {packet_name}::{func_name}(const unsigned int id) {{\n",
            f"\tif (id != 2 && id != 3) throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id\");\n"
            f"\tmilhoja::Stream stream = id == 2 ? stream2_ : stream3_;\n",
            f"\tif(!stream.isValid()) throw std::logic_error(\"[{packet_name}::{func_name}] Queue \" + std::to_string(id) + \" is not valid.\");\n",
            f"\treturn stream.accAsyncQueue;\n"
            f"}}\n"
        ])

        # release extra queue
        func_name = "releaseExtraQueue"
        file.writelines([
            f"void {packet_name}::{func_name}(const unsigned int id) {{\n",
            f"\tif (id != 2 && id != 3) throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id\");\n",
            f"\tmilhoja::Stream stream = id == 2 ? stream2_ : stream3_;\n",
            f"\tif (!stream.isValid()) throw std::logic_error(\"[{packet_name}::{func_name}] Queue \" + std::to_string(id) + \" is not valid.\");\n",
            f"\tmilhoja::RuntimeBackend::instance().releaseStream(stream);\n"
            f"}}\n"
        ])


    name = parameters["name"]
    with open(name + ".cpp", "w") as code:
        code.write(GENERATED_CODE_MESSAGE)
        # Most of the necessary includes are included in the datapacket header file.
        code.write(f"#include \"{name}.h\"\n")
        code.write(f"#include <cassert>\n") # do we need certain includes?
        code.write(f"#include <cstring>\n") # for advancing by 1 byte (char *)
        code.write(f"#include <stdexcept>\n")
        code.write(f"#include <Milhoja.h>\n")
        # This is assuming all of the types being passed are milhoja types

        if "problem" in parameters: code.write( "#include \"%s.h\"\n" % parameters["problem"] )
        code.write("#include <Driver.h>\n")

        generate_constructor(code, parameters)   
        generate_destructor(code, parameters) 
        generate_unpack(code, parameters)
        generate_pack(code, parameters)
        generate_release_queues(code, parameters)
        generate_clone(code, parameters)

# Creates a header file based on the given parameters.
# Lots of boilerplate and class member generation.
# TODO: Pack and unpack functions use a specific variable involving CC1 and CC2. How can we specify this in tile-in and tile-out?
def generate_header_file(parameters):

    # Every packet json should have a name associated with it.
    if "name" not in parameters or not isinstance(parameters["name"], str):
        raise RuntimeError("Packet does not include a name.")

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
        header.write(indent + f"{name}(Real* dt = nullptr);\n")
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
        if GENERAL in parameters:
            known_sections.add(GENERAL)
            general = parameters[GENERAL] # general section is the general copy in data information
            for item in general:
                block_size_var = f"{item}{BLOCK_SIZE}"
                header.write(f"{indent}milhoja::{general[item]}* {item};\n")    # add a new variable for each item
                header.write(f"{indent}{SIZE_T} {block_size_var};\n")
                vars_and_types[block_size_var] = f"milhoja::{general[item]}"

        # Generate private variables. We need to create a variable
        # for all necessary sizes

        if T_MDATA in parameters:
            known_sections.add(T_MDATA)
            for item in parameters[T_MDATA]:
                new_variable = f"{item}{BLOCK_SIZE}"
                header.write(f"{indent}{SIZE_T} {new_variable};\n")
                vars_and_types[new_variable] = SIZE_T

        if T_IN in parameters:
            known_sections.add(T_IN)
            for item in parameters[T_IN]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        if T_OUT in parameters:
            known_sections.add(T_OUT)
            for item in parameters[T_OUT]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        if T_OUT in parameters:
            known_sections.add(T_OUT)
            for item in parameters[T_OUT]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        level -= 1
        indent = get_indentation(level)
        header.write("};\n")

        # end start define
        header.write("#endif\n")

    return

def generate_packet_from_json(file):
    with open(file, "r") as file:
        data = json.load(file)
        generate_header_file(data)
        generate_cpp_file(data)

    return

if __name__ == "__main__":
    #Check if some file path was passed in
    if len(sys.argv) < 2:
        print("Usage: python packet_generator.py [data_file]")
        
    generate_packet_from_json(sys.argv[1])