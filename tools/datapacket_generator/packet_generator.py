# Author: Wesley Kwiecinski

# Packet generator for Milhoja. Script takes in a 
# JSON file and generates cpp code for a data packet
# based on the contents of the json file.
# 
# TODO: We should also be adding support for generating fortran file packets in the future.
# 
# TODO: Work on documentation for packet json format. Refer to DataPacketGeneratorDoc for documentation
# 
# TODO: Add nTiles and PacketContents to general / non-tile specific data section on packet generation startup.
# 
# TODO: Restructure pack generation into 2 phases: pointer determination & copy data phases.

import sys
import json
import argparse
import milhoja_data as mdata

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

datatype_to_include_map = {
    "IntVect": ""
}

# All possible keys.
EXTRA_STREAMS = 'n-extra-streams'
GENERAL = "general"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_IN_OUT = "tile-in-out"
T_OUT = "tile-out"
# 

# type constants / naming keys.
NEW = "new_"
SIZE_T = "std::size_t"
BLOCK_SIZE = "_BLOCK_SIZE_HELPER"
N_TILES = "nTiles"
SCRATCH = "_scratch_d"
TILE_DESC = "tileDesc_h"
DATA_P = "_data_p"
DATA_D = "_data_d"
OUT = "out"
START_P = "_start_p"
START_D = "_start_d"
PTRS = "pointers"
PINDEX = f"{PTRS}_index"
# 

# TODO: Variant 2 uses an array of extra queues instead of just using 2 extra ones if dim is 3.
# Is there a way we can specify that in the json packet? Is it worth it?

# these might not be necessary
vars_and_types = {}
level = 0

# TODO: Maybe we can create dictionaries derived from sections in the json? Something to think about
device_array_pointers = {}
nstreams = 1

all_pointers = set()
types = set()
includes = set()

# TODO: It might be beneficial to write helper methods or a wrapper class for files
# to help with consistently writing to the file, so we
# don't have to put {indent} or \n in every line we write.\

def is_enumerable_type(var):
    return isinstance(var, (dict, list))

def generate_fortran_header_file(parameters):
    ...

def generate_fortran_code_file(parameters):
    ...

def generate_cpp_code_file(parameters):
    def generate_constructor(file, params):
            # function definition
            file.write(f"{params['name']}::{params['name']}(const milhoja::Real {NEW}dt) : milhoja::DataPacket{{}}, \n")
            extra_streams = params[EXTRA_STREAMS]
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
            # TODO: We can just use dict.get instead of storing the known sections.
            for section in params:
                if isinstance(params[section], dict):
                    for item in params[section]:
                        if not isinstance(params[section][item], (dict, list)):
                            file.write(f"{indent}{item}{BLOCK_SIZE} = sizeof({params[section][item]});\n")
                        else:
                            extents, nunkvar, empty = mdata.parse_extents(params[section][item]['extents'], params[section][item]['type'])
                            file.write(f"{indent}{item}{BLOCK_SIZE} = {extents};\n")
            
            # TODO: What if we need to add other variables?
            # We need to add them to the constructor args.
            for item in params.get(GENERAL, {}):
                file.write(f"{indent}{item} = {NEW}{item};\n")

            file.write("}\n\n")

    # TODO: Eventually come back and fix the streams to use the array implementation.
    def generate_destructor(file, params):
        packet_name = params["name"]
        extra_streams = params.get(EXTRA_STREAMS, 0)
        file.write(f"{packet_name}::~{packet_name}(void) {{\n")
        indent = '\t'
        # if params["ndim"] == 3:
        #     file.write(f"{indent}if (stream2_.isValid() || stream3_.isValid()) throw std::logic_error(\"[DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3] One or more extra streams not released\");")
        # file.write("#if MILHOJA_NDIM==3\n")
        for i in range(2, extra_streams+2):
            file.write(f"{indent}if (stream{i}_.isValid()) throw std::logic_error(\"[DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3] One or more extra streams not released\");\n")
        # file.write("#endif\n")
        # if extra_streams > 0:
        #     file.writelines([
        #         f"{indent}for (unsigned int i=0; i < EXTRA_STREAMS; ++i)\n",
        #         f"{indent*2}if (streams_[i].isValid()) \n",
        #         f"{indent*3}throw std::logic_error(\"[{packet_name}::~{packet_name}] One or more extra streams not released.\");\n"
        #     ])
        file.write(f"{indent}nullify();\n")
        file.write(f"}}\n\n")
        return

    # TODO: Some parts of the unpack method need to be more generalized.
    def generate_unpack(file, params):
        packet_name = params["name"]
        func_name = "unpack"
        level = 1
        indent = level * "\t"
        file.write(f"void {packet_name}::unpack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        file.writelines([
            f"{indent}if (tiles_.size() <= 0) throw std::logic_error(\"[{packet_name}::{func_name}] Empty data packet.\");\n",
            f"{indent}if (!stream_.isValid()) throw std::logic_error(\"[{packet_name}::{func_name}] Stream not acquired.\");\n",
            f"{indent}if (pinnedPtrs_ == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] No pinned pointers set.\");\n"
            f"{indent}if ( startVariable_ < UNK_VARS_BEGIN || startVariable_ > UNK_VARS_BEGIN || endVariable_ < UNK_VARS_BEGIN || endVariable_ > UNK_VARS_END )\n"
            f"{indent}{indent}throw std::logic_error(\"[{packet_name}::{func_name}] Invalid variable mask\");\n"
            f"{indent}RuntimeBackend::instance().releaseStream(stream_);\n"
            f"{indent}assert(!stream_.isValid());\n\n"
            f"{indent}for ({SIZE_T} n=0; n < tiles_.size(); ++n) {{\n"
        ])

        indent = 2 * '\t'

        # TODO: CC location remains consistent across data packets?
        # Also location_ gets modified outside of the data packet
        # so this stays for now
        file.writelines([
            f"{indent}Tile* tileDesc_h = tiles_[n].get();\n",
            f"{indent}Real* data_h = tileDesc_h->dataPtr();\n",
            f"{indent}const Real* data_p = nullptr;\n",
            f"{indent}switch (location_) {{\n",
            f"{indent}\tcase PacketDataLocation::CC1: data_p = pinnedPtrs_[n].CC1_data; break;\n",
            f"{indent}\tcase PacketDataLocation::CC2: data_p = pinnedPtrs_[n].CC2_data; break;\n",
            f"{indent}\tdefault: throw std::logic_error(\"[{packet_name}::{func_name}] Data not in CC1 or CC2.\");\n"
            f"{indent}}}\n"
        ])

        # data_h and data_p checks
        file.writelines([
            f"{indent}if (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid pointer to data in host memory.\");\n",
            f"{indent}if (data_p == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Invalid pointer to data in pinned memory.\");\n"
        ])

        # TODO: Should we use nunkvars from tile-in?
        file.writelines([
            f"{indent}assert(UNK_VARS_BEGIN == 0);\n"
            f"{indent}assert(UNK_VARS_END == NUNKVAR - 1);\n"
        ])

        dict_to_use = {}
        if params.get(T_IN_OUT): dict_to_use = params.get(T_IN_OUT)
        elif params.get(T_OUT): dict_to_use = params.get(T_OUT)

        # TODO: this only uses the first item in tile-in or tile-out. we need to adjust it to use all array types specified in
        # either.
        if dict_to_use:
            it = iter(dict_to_use)
            item = next(it)
            data_type = dict_to_use[item]['type']
            extents, nunkvars, empty = mdata.parse_extents(dict_to_use[item]['extents'])
            # TODO: The way the constructor and header files we need to do some division to 
            # get the origin num vars per CC per variable. This is a way to do it without creating
            # another variable. Low priority
            # num_elems_per_cc_per_var = ' * '.join(dict_to_use[item]['extents'][:-1])
            num_elems_per_cc_per_var = f'({item}{BLOCK_SIZE} / ( ({nunkvars}) * sizeof({data_type})) )'
            file.writelines([
                f"{indent}{SIZE_T} offset = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>(startVariable_);\n",
                f"{indent}{data_type}* start_h = data_h + offset;\n"
                f"{indent}const {data_type}* start_p = data_p + offset;\n"
            ])
            num_elems_per_cc_per_var = f'(({item}{BLOCK_SIZE}) / ({nunkvars}))'
            file.writelines([
                f"{indent}{SIZE_T} nBytes = (endVariable_ - startVariable_ + 1) * ({num_elems_per_cc_per_var});\n"
                f"{indent}std::memcpy((void*)start_h, (void*)start_p, nBytes);\n"                
            ])

        
        indent = '\t'
        file.write(f"{indent}}}\n")

        file.write(f"}}\n\n")
        return

    # TODO: Improve pack generation code
    # TODO: {N_TILES} are a guaranteed part of general, same with PacketContents. We also want to consider letting the user add numeric constants to general
    # TODO: We should have constants for variable names in the cpp file generation so they can be easily changed.
    def generate_pack(file, params):
        packet_name = params["name"]
        ndim = params["ndim"]
        func_name = "pack"
        level = 1
        indent = level * "\t"
        tab = "\t"
        file.write(f"void {packet_name}::pack(void) {{\n")
        file.write(f"{indent}using namespace milhoja;\n")
        
        # Error checking
        file.writelines([
            f"{indent}std::string errMsg = isNull();\n",
            f"{indent}if (errMsg != \"\")\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent}else if (tiles_.size() == 0)\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] No tiles added.\");\n"
            f"{indent}\n"
            f"{indent}Grid& grid = Grid::instance();\n\n"
        ])

        # SIZE DETERMINATION SECTION
        file.write(f"{indent}/// SIZE DETERMINATION\n")
        # # Scratch section generation.
        file.write(f"{indent}// Scratch section\n")

        bytesToGpu = set()
        returnToHost = set()
        bytesPerPacket = set()

        nScratchArrs = 0
        file.write(f"{indent}{SIZE_T} nScratchPerTileBytes = 0")
        for var in params.get(T_SCRATCH, []):
            nScratchArrs += 1
            file.write(f" + {var}{BLOCK_SIZE}")
        file.write(f";\n{indent}unsigned int nScratchArrays = {nScratchArrs};\n")
        bytesPerPacket.add("(nTiles * nScratchPerTileBytes)")

        # # Copy-in section generation.
        # Non tile specific data
        file.write(f"\n{indent}// non tile specific data\n")
        file.write(f"{indent}{SIZE_T} {N_TILES} = tiles_.size();\n")
        file.write(f"{indent}{SIZE_T} nCopyInBytes = sizeof({SIZE_T}) ")
        for item in params.get(GENERAL, []):
            file.write(f"+ {item}{BLOCK_SIZE} ")
        file.write(f"+ {N_TILES} * sizeof(PacketContents);\n") # we can eventually get rid of packet contents so this will have to change.
        bytesToGpu.add("nCopyInBytes")
        bytesPerPacket.add("nCopyInBytes")

        number_of_arrays = len(params.get(T_IN, {})) + len(params.get(T_IN_OUT, {})) + len(params.get(T_OUT, {}))
        # TODO: If we want to allow any specification for dimensionality of arrays we need to change this
        file.write(f"{indent}{SIZE_T} nBlockMetadataPerTileBytes = (nScratchArrays + {number_of_arrays}) * sizeof(FArray4D)")
        for item in params.get(T_MDATA, []):
            file.write(f" + {item}{BLOCK_SIZE}")
        file.write(f";\n")
        bytesToGpu.add("(nTiles * nBlockMetadataPerTileBytes)")
        bytesPerPacket.add("(nTiles * nBlockMetadataPerTileBytes)")

        # copy in data
        cin = params.get(T_IN, {})
        file.write(f"{indent}{SIZE_T} nCopyInDataPerTileBytes = 0")
        if cin:
            for item in cin:
                file.write(f" + {item}{BLOCK_SIZE}")
            bytesToGpu.add("(nTiles * nCopyInDataPerTileBytes)")
            bytesPerPacket.add("(nTiles * nCopyInDataPerTileBytes)")
        file.write(f";\n")

        # copy in out data
        cinout = params.get(T_IN_OUT, {})
        file.write(f"{indent}{SIZE_T} nCopyInOutDataPerTileBytes = 0")
        if cinout:
            for item in cinout:
                file.write(f" + {item}{BLOCK_SIZE}")
            bytesToGpu.add("(nTiles * nCopyInOutDataPerTileBytes)")
            returnToHost.add("(nTiles * nCopyInOutDataPerTileBytes)")
            bytesPerPacket.add("(nTiles * nCopyInOutDataPerTileBytes)")
        file.write(f";\n")

        # copy out
        cout = params.get(T_OUT, {})
        file.write(f"{indent}{SIZE_T} nCopyOutDataPerTileBytes = 0")
        if cout:
            for item in cout:
                file.write(f" + {item}{BLOCK_SIZE}")
            # bytesToGpu.add("(nTiles * nCopyOutDataPerTileBytes)")
            returnToHost.add("(nTiles * nCopyOutDataPerTileBytes)")
            bytesPerPacket.add("(nTiles * nCopyOutDataPerTileBytes)")
        file.write(f";\n")

        # 
        file.writelines([
            f"{indent}// Copy out section\n",
            f"{indent}nCopyToGpuBytes_ = {' + '.join(bytesToGpu)};\n",
            f"{indent}nReturnToHostBytes_ = {' + '.join(returnToHost)};\n",
            f"{indent}{SIZE_T} nBytesPerPacket = {' + '.join(bytesPerPacket)};\n"
        ])

        # # acquire gpu mem
        # Note that copyin, copyinout, and copyout all have a default of 0. So to keep code generation easy
        # we set them to the default of 0 even if they don't exist in the json.
        file.write(f"{indent}RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - {N_TILES} * nScratchPerTileBytes, &packet_p_, nBytesPerPacket, &packet_d_);\n")
        file.write(f"{indent}/// END\n\n")

        file.write(f"{indent}/// POINTER DETERMINATION\n\n")
        # array to store all pointers to copy in later.
        # num_of_arrays = f"1 + {len(params.get(GENERAL, {}))} + ({N_TILES} * { len(params.get(T_SCRATCH, {})) + len(params.get(T_MDATA, {})) + len(params.get(T_IN, {})) + len(params.get(T_IN_OUT)) + len(params.get(T_OUT, {})) })"
        file.writelines([
            f"{indent}copyargs.clear();\n\n"
        ])
        
        file.writelines([
            f"{indent}location_ = PacketDataLocation::CC1;\n" # TODO: We need to change this
            f"{indent}char* scratchStart_d = static_cast<char*>(packet_d_);\n",
            f"{indent}copyInStart_p_ = static_cast<char*>(packet_p_);\n",
            f"{indent}copyInStart_d_ = scratchStart_d + {N_TILES} * nScratchPerTileBytes;\n",
            f"{indent}copyInOutStart_p_ = copyInStart_p_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes) + ({N_TILES} * nCopyInDataPerTileBytes);\n",
            f"{indent}copyInOutStart_d_ = copyInStart_d_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes) + ({N_TILES} * nCopyInDataPerTileBytes);\n",
        ])

        if T_OUT in params:
            file.writelines([
                f"{indent}char* copyOutStart_p = copyInOutStart_p_;\n",
                f"{indent}char* copyOutStart_d = copyInOutStart_d_;\n"
            ])

        file.writelines([
            f"{indent}if (pinnedPtrs_) throw std::logic_error(\"{packet_name}::pack Pinned pointers already exist\");\n",
            f"{indent}pinnedPtrs_ = new BlockPointersPinned[{N_TILES}];\n"
        ]) 

        # Scratch section does not get transferred from gpu

        # copy-in section
        file.writelines([
            f"{indent}static_assert(sizeof(char) == 1, \"Invalid char size\");\n", # we might not need this anymore
            f"{indent}char* ptr_p = copyInStart_p_;\n",
            f"{indent}char* ptr_d = copyInStart_d_;\n\n",
        ])

        # automatically generate nTiles data PacketContents
        file.writelines([
            f"{indent}std::memcpy((void*)ptr_p, (void*)&{N_TILES}, sizeof({SIZE_T}));\n",
            f"//{indent}copyargs.push_back( {{ (void*)&{N_TILES}, (void*)ptr_p, sizeof({SIZE_T}) }} );\n"
            f"{indent}ptr_p += sizeof({SIZE_T});\n",
            f"{indent}ptr_d += sizeof({SIZE_T});\n\n"
        ])

        for item in params.get(GENERAL, []):
            file.writelines([
                f"//{indent}copyargs.push_back( {{ (void*)&{item}, (void*)ptr_p, sizeof({item}{BLOCK_SIZE}) }} );\n"
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"{indent}ptr_p += sizeof({item}{BLOCK_SIZE});\n",
                f"{indent}ptr_d += sizeof({item}{BLOCK_SIZE});\n\n"
            ])

        # packet contents comes after general section in hydro variants.
        file.writelines([
            f"{indent}contents_p_ = static_cast<PacketContents*>((void*)ptr_p);\n",
            f"{indent}contents_d_ = static_cast<PacketContents*>((void*)ptr_d);\n",
            f"{indent}ptr_p += {N_TILES} * sizeof(PacketContents);\n",
            f"{indent}ptr_d += {N_TILES} * sizeof(PacketContents);\n\n"
        ])

        for idx,item in enumerate(params.get(T_IN, {})):
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyInStart_p_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
                file.write(f"{indent}char* {item}{START_D} = copyInStart_d_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            else:
                l = list(params[T_IN])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {l[idx-1]}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_D} + {l[idx-1]}{BLOCK_SIZE};\n")

        # TODO: We don't need to worry about data location since generated code should automatically know the data location.
        # TODO: The variants for each packet don't have CC1 set co copyInOut.
        # TODO: When do we change where the start of CC1 and CC2 data is located?
        # for item in params.get(T_IN_OUT, {}):
        # TODO: We need to change this, T_OUT, and T_IN to work like the scratch section
        for idx,item in enumerate(params.get(T_IN_OUT, {})):
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyInOutStart_p_;\n")#+ nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
                file.write(f"{indent}char* {item}{START_D} = copyInOutStart_d_;\n")# + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            else:
                l = list(params[T_IN_OUT])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {item}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_D} + {item}{BLOCK_SIZE};\n")
         
        for idx,item in enumerate(params.get(T_OUT, {})):
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyOutStart_p;\n")# + {N_TILES} * copyInOutDataPerTileBytes;\n")
                file.write(f"{indent}char* {item}{START_D} = copyOutStart_d;\n")# + {N_TILES} * copyInOutDataPerTileBytes;\n")
            else:
                l = list(params[T_OUT])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {item}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_D} + {item}{BLOCK_SIZE};\n")

        # Create all scratch ptrs.
        # scr = sorted(list(params.get(T_SCRATCH, {}).keys()))
        scr = list(params.get(T_SCRATCH, {}).keys())
        for idx,item in enumerate(scr):
            if idx == 0:  # we can probably use an iterator here instead
                file.write(f"{indent}char* {scr[idx]}{START_D} = scratchStart_d;\n")
            else:
                file.write(f"{indent}char* {scr[idx]}{START_D} = {scr[idx-1]}{START_D} + {scr[idx-1]}{BLOCK_SIZE};\n")
        file.write(f"{indent}PacketContents* tilePtrs_p = contents_p_;\n")
            
        # tile specific metadata.
        file.write(f"{indent}for ({SIZE_T} n=0; n < {N_TILES}; ++n, ++tilePtrs_p) {{\n")
        indent = "\t" * 2
        # TODO: We need to change this to use the actual types specified in the json when we remove PacketContents.
        file.writelines([
            f"{indent}Tile* tileDesc_h = tiles_[n].get();\n",
            f"{indent}if (tileDesc_h == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Bad tileDesc.\");\n",
            f"{indent}const RealVect deltas = {TILE_DESC}->deltas();\n"
            f"{indent}const IntVect lo = {TILE_DESC}->lo();\n"
            f"{indent}const IntVect hi = {TILE_DESC}->hi();\n"
            f"{indent}const IntVect loGC = {TILE_DESC}->loGC();\n"
            f"{indent}const IntVect hiGC = {TILE_DESC}->hiGC();\n"
            f"{indent}Real* data_h = {TILE_DESC}->dataPtr();\n"
            f"{indent}if (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid ptr to data in host memory.\");\n\n"
        ])

        # TODO: THis code here is a problem if we want to have any number of arrays in each section.
        if T_IN in params:
            size = "0 + " + ' + '.join( f'{item}{BLOCK_SIZE}' for item in params.get(T_IN, {}) )
            file.write(f"{indent}std::memcpy((void*){'_'.join(params[T_IN])}{START_P}, (void*)data_h, {size}\n")
            file.write(f"//{indent}copyargs.push_back( {{ (void*){'_'.join(params[T_IN])}{START_P}, (void*)ptr_p, {size} }} );\n")
        elif T_IN_OUT in params:
            size = "0 + " + ' + '.join( f'{item}{BLOCK_SIZE}' for item in params.get(T_IN_OUT, {}) )
            file.write(f"{indent}std::memcpy((void*){'_'.join(params[T_IN_OUT])}{START_P}, (void*)data_h, {size}\n")
            file.write(f"//{indent}copyargs.push_back( {{ (void*){'_'.join(params[T_IN_OUT])}{START_P}, (void*)ptr_p, {size} }} );\n")
        # file.write(f");\n")

        # be careful here, is pinnedptrs tied to tile-in-out or tile-out? What about tile-in?
        # We need to change this so that we aren't accidentally assigning CC1_data to a cc2 ptr.
        # TODO: We need to change this code to work with multiple array types in each section of the json
        if T_IN_OUT in params:
            nxt = next(iter(params[T_IN_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = static_cast<{params[T_IN_OUT][nxt]['type']}*>((void*){ '_'.join(params[T_IN_OUT].keys()) }{START_P});\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = nullptr;\n")

        if T_OUT in params:
            nxt = next(iter(params[T_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = static_cast<{params[T_OUT][nxt]['type']}*>((void*){ '_'.join(params[T_OUT].keys()) }{START_P});\n\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = nullptr;\n\n")

        possible_tile_ptrs = ['deltas', 'lo', 'hi', 'CC1', 'CC2', 'FCX', 'FCY', 'FCZ'] # we will eventually completely get rid of this.
        # Add metadata to ptr
        for item in params.get(T_MDATA, []):
            possible_tile_ptrs.remove(item)
            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<{params[T_MDATA][item]}*>((void*)ptr_d);\n",
                f"//{indent}copyargs.push_back( {{ (void*)&{item}, (void*)ptr_p, {item}{BLOCK_SIZE} }} );\n"
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"{indent}ptr_p += {item}{BLOCK_SIZE};\n"
                f"{indent}ptr_d += {item}{BLOCK_SIZE};\n\n"
            ])

        # TODO: This is not going to work for generic names for arrays. We need to 
        # add specifications to the json file.
        for item in device_array_pointers:
            d = 4 # assume d = 4 for now.
            section = device_array_pointers[item]['section']
            type = device_array_pointers[item]['type']
            extents, nunkvars, indexer = mdata.parse_extents(device_array_pointers[item]['extents'])
            c_args = mdata.constructor_args[indexer]

            file.write(f"{indent}tilePtrs_p->{item}_d = static_cast<FArray{d}D*>((void*)ptr_d);\n")
            file.writelines([
                f"{indent}FArray{d}D {item}_d{{ static_cast<{type}*>((void*){item}{START_D}), {c_args}, {nunkvars}}};\n"
                f"//{indent}copyargs.push_back( {{ (void*)&{item}_d, (void*)ptr_p, sizeof(FArray{d}D) }} );\n"
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}_d, sizeof(FArray{d}D));\n",
                f"{indent}ptr_p += sizeof(FArray{d}D);\n",
                f"{indent}ptr_d += sizeof(FArray{d}D);\n",
            ])

            if section == T_SCRATCH:
                file.write(f"{indent}{item}{START_D} += nScratchPerTileBytes;\n\n")
            else:
                file.write(f"{indent}{item}{START_P} += {item}{BLOCK_SIZE};\n")
                file.write(f"{indent}{item}{START_D} += {item}{BLOCK_SIZE};\n\n")

            possible_tile_ptrs.remove(item)

        # if there are unremoved items we set them to nullptr.
        # TODO: This needs to change when removing PacketContents
        for item in possible_tile_ptrs:
            file.write(f"{indent}tilePtrs_p->{item}_d = nullptr;\n")

        indent = "\t"

        file.write(f"{indent}}}\n")
        file.write(f"{indent}/// END\n\n")
        
        file.write(f"{indent}/// COPY INTO GPU MEMORY\n\n")
        # file.writelines([
        #     f"{indent}for(int i = 0; i < number_of_pointers; ++i) {{\n",
        #     f"{indent*2}std::memcpy({PTRS}[i].destination, {PTRS}[i].source, {PTRS}[i].size);\n"
        #     f"{indent}}}\n\n"
        # ])
        file.write(f"{indent}/// END\n\n")

        # request stream at end
        file.writelines([
            f"{indent}stream_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}if (!stream_.isValid()) {{\n",
            f"{indent * 2}throw std::runtime_error(\"[{packet_name}::pack] Unable to acquire stream\");\n{indent}}}\n"
        ])

        # TODO: CHange to use the array implementation.
        e_streams = params.get(EXTRA_STREAMS, 0)
        if e_streams > 0:
            # file.write("#if MILHOJA_NDIM==3\n")
            for i in range(2, e_streams+2):
                file.writelines([
                    f"{indent}stream{i}_ = RuntimeBackend::instance().requestStream(true);\n",
                    f"{indent}if (!stream{i}_.isValid()) throw std::runtime_error(\"[{packet_name}::{func_name}] Unable to acquire extra stream.\");\n"
                ])
            # file.write("#endif\n")
            # file.writelines([
            #     f"{indent}for (unsigned int i=0; i < EXTRA_STREAMS; ++i) {{\n",
            #     f"{indent*2}streams_[i] = RuntimeBackend::instance().requestStream(true);\n",
            #     f"{indent*2}if (!streams_[i].isValid()) throw std::runtime_error(\"[{packet_name}::{func_name}] Unable to acquire extra stream.\"); \n"
            #     f"{indent}}}\n"
            # ])


            # file.writelines([
            #     f"# if MILHOJA_NDIM == 3\n", # we can get rid of the compiler directives eventually.
            #     f"{indent}stream2_ = RuntimeBackend::instance().requestStream(true);\n",
            #     f"{indent}stream3_ = RuntimeBackend::instance().requestStream(true);\n",
            #     f"{indent}if (!stream2_.isValid() || !stream3_.isValid()) {{\n",
            #     f"{indent * 2}throw std::runtime_error(\" [{packet_name}::pack] Unable to acquire extra streams\");\n{indent}}}\n#endif\n"
            # ])
        
        file.write(f"}}\n\n")
        return
    
    # Generate clone method
    def generate_clone(file, params):
        packet_name = params["name"]
        file.writelines([
            f"std::unique_ptr<milhoja::DataPacket> {packet_name}::clone(void) const {{\n",
            f"\treturn std::unique_ptr<milhoja::DataPacket>{{ new {packet_name}{{dt}} }};\n"
            f"}}\n\n"
        ])

    # TODO: Right now the code generates stream{n}_ for n extra streams.
    #       Ideally, we would store all of those in an array and use indexing to check the specific id.
    #       This is what hydro variant2 actually does, but since variant 1 and variant 3 use this system,
    #       I will manually generate and check all streams until we can write our own task functions,
    #       since releasing extra queues is called outside of the datapacket object.
    def generate_release_queues(file, params):
        packet_name = params['name']
        extra_streams = params.get(EXTRA_STREAMS, 0)
        indent = '\t'
        # extra async queues
        # only generate extra functions if we have more than 1 stream
        if extra_streams > 0:
            # get extra async queue
            func_name = "extraAsynchronousQueue"
            # file.write("#if MILHOJA_NDIM==3\n")
            file.writelines([
                f"int {packet_name}::{func_name}(const unsigned int id) {{\n",
                f"{indent}if ((id < 2) || (id > EXTRA_STREAMS + 1))\n"
                f"{indent*2}throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id.\");\n"
                # f"{indent}if (!streams_[id-2].isValid())\n"
                # f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] Extra queue invalid.\");\n"
                # f"{indent}return streams_[id-2].accAsyncQueue;\n"
                # f"}}\n\n"
            ])

            file.write(f"{indent}switch(id) {{\n")
            for i in range(2, extra_streams+2):
                file.writelines([
                    f"{indent * 2}case {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error(\"[{packet_name}::{func_name}] Extra queue invalid. ({i})\"); }} return stream{i}_.accAsyncQueue;\n"
                ])
            file.write(f"{indent}}}\n{indent}return 0;\n}}\n\n")

            # Release extra queue
            func_name = "releaseExtraQueue"
            file.writelines([
                f"void {packet_name}::{func_name}(const unsigned int id) {{\n",
                f"{indent}if ((id < 2) || (id > EXTRA_STREAMS + 1))\n"
                f"{indent*2}throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id.\");\n"
                # f"{indent}if (!streams_[id-2].isValid())\n"
                # f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] Extra queue invalid.\");\n"
                # f"{indent}milhoja::RuntimeBackend::instance().releaseStream(streams_[id-2]);\n"
                # f"}}\n\n"
            ])

            file.write(f"{indent}switch(id) {{\n")
            for i in range(2, extra_streams+2):
                file.writelines([
                    f"{indent * 2}case {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error(\"[{packet_name}::{func_name}] Extra queue invalid. ({i})\"); }} milhoja::RuntimeBackend::instance().releaseStream(stream{i}_);\n"
                ])
            file.write(f"{indent}}}\n}}\n\n")
            # file.write("#endif\n")

    if not parameters:
        raise ValueError("Parameters is empty or null.")
    
    # TODO: We should be generating the include files using a map based on all different types given in the json packet. 
    name = parameters["name"]
    ndim = parameters["ndim"] # TODO: should we really force the user to specify the number of dims in the json packet?
    with open(name + ".cpp", "w") as code:
        # We might need to include specific headers based on the contents of the json packet
        code.write(GENERATED_CODE_MESSAGE)
        # Most of the necessary includes are included in the datapacket header file.
        code.write(f"#include \"{name}.h\"\n")
        code.write(f"#include <cassert>\n") # do we need certain includes?
        code.write(f"#include <cstring>\n") # for advancing by 1 byte (char *)
        code.write(f"#include <stdexcept>\n")
        code.write(f"#include <Milhoja_Grid.h>\n")
        code.write(f"#include <Milhoja_RuntimeBackend.h>\n")
        # This is assuming all of the types being passed are milhoja types

        if "problem" in parameters: code.write( "#include \"%s.h\"\n" % parameters["problem"] )
        code.write("#include \"Driver.h\"\n")

        generate_constructor(code, parameters)   
        generate_destructor(code, parameters)
        generate_release_queues(code, parameters)
        # generate clone method
        generate_clone(code, parameters)
        generate_pack(code, parameters)
        generate_unpack(code, parameters)

# Creates a header file based on the given parameters.
# Lots of boilerplate and class member generation.
# TODO: Pack and unpack functions use a specific variable involving CC1 and CC2. How can we specify this in tile-in and tile-out?
# TODO: Should the user have to specify the molhoja dim in the json file?
# TODO: Get all necessary include statements from JSON file.
def generate_cpp_header_file(parameters):
    if not parameters:
        raise ValueError("Parameters is null.")

    # Every packet json should have a name associated with it.
    if "name" not in parameters or not isinstance(parameters["name"], str):
        raise RuntimeError("Packet does not include a name.")

    with open(parameters["name"] + ".h", "w") as header:
        name = parameters["name"]
        extra_streams = parameters.get(EXTRA_STREAMS, 0)
        defined = name.upper()
        level = 0
        private_variables = []
        header.write(GENERATED_CODE_MESSAGE)
        
        # define statements
        header.write(f"#ifndef {defined}\n")
        header.write(f"#define {defined}\n")
        # boilerplate includes
        header.write("#include <Milhoja.h>\n")
        header.write("#include <Milhoja_DataPacket.h>\n")

        # Everything in the packet consists of pointers to byte regions
        # so we make every variable a pointer
        # TODO: What if we want to put array types in any section?
        # TODO: Create a helper function that checks if the item is a string/scalar or an array type
        #       to perform certain functions 
        # TODO: Assume FArray4D by default for now.
        if GENERAL in parameters:
            general = parameters[GENERAL] # general section is the general copy in data information
            for item in general:
                block_size_var = f"{item}{BLOCK_SIZE}"
                var = f"\tmilhoja::{general[item]} {item};\n"
                size_var = f"\t{SIZE_T} {block_size_var};\n"
                is_enumerable = is_enumerable_type(general[item])
                types.add(general[item] if not is_enumerable else general[item]['type'])
                if is_enumerable: types.add( f"FArray4D" )
                # header.write(f"{indent}milhoja::{general[item]} {item};\n")    # add a new variable for each item
                # header.write(f"{indent}{SIZE_T} {block_size_var};\n")
                # header.write(f"#include {includes[ general[item] ]}\n")
                private_variables.append(var)
                private_variables.append(size_var)
                vars_and_types[block_size_var] = f"milhoja::{general[item]}"

        # Generate private variables for each section. Here we are creating a size helper
        # variable for each item in each section based on the name of the item.

        # TODO: We can scrunch all of these separate loops into one once we remove the very specific code for
        #       the CC1, CC2, Scratch pointer section.
        if T_MDATA in parameters:
            for item in parameters[T_MDATA]:
                new_variable = f"{item}{BLOCK_SIZE}"
                is_enumerable = is_enumerable_type(parameters[T_MDATA][item])
                types.add(parameters[T_MDATA][item] if not is_enumerable_type(parameters[T_MDATA][item]) else parameters[T_MDATA][item]['type'])
                if is_enumerable: types.add( f"FArray4D" )
                private_variables.append(f"\t{SIZE_T} {new_variable};\n")
                vars_and_types[new_variable] = SIZE_T

        for sect in [T_IN, T_IN_OUT, T_OUT, T_SCRATCH]:
            for item in parameters.get(sect, {}):
                private_variables.append(f"\t{SIZE_T} {item}{BLOCK_SIZE};\n")
                is_enumerable = is_enumerable_type(parameters[sect][item])
                types.add(parameters[sect][item] if not is_enumerable_type(parameters[sect][item]) else parameters[sect][item]['type'])
                if is_enumerable: types.add( f"FArray4D" )
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T
                device_array_pointers[item] = {"section": sect, **parameters[sect][item]}

        # we only want to include things if they are found in the include dict.
        header.write( ''.join( f"#include {mdata.imap[item]}\t\n" for item in types if item in mdata.imap) )
        header.write("#include <vector>\n")

        header.writelines([
            f"\nstruct MemCopyArgs {{\n",
            f"\tvoid* src = nullptr;\n",
            f"\tvoid* dest = nullptr;\n",
            f"\t{SIZE_T} size = 0;\n"
            f"}};\n\n"
        ])

        # class definition
        header.write(f"class {name} : public milhoja::DataPacket {{ \n")
        level += 1
        indent = '\t' * level

        # public information
        header.write("public:\n")
        header.write(f"{indent}std::unique_ptr<milhoja::DataPacket> clone(void) const override;\n")
        header.write(indent + f"{name}(const milhoja::Real {NEW}dt);\n")
        header.write(indent + f"~{name}(void);\n")

        # Constructors & = operations
        header.writelines([\
            f"{indent}{name}({name}&)                  = delete;\n",
            f"{indent}{name}(const {name}&)            = delete;\n",
            f"{indent}{name}({name}&& packet)          = delete;\n",
            f"{indent}{name}& operator=({name}&)       = delete;\n",
            f"{indent}{name}& operator=(const {name}&) = delete;\n",
            f"{indent}{name}& operator=({name}&& rhs)  = delete;\n"
        ])

        # pack & unpack methods
        header.writelines([
            f"{indent}void pack(void) override;\n",
            f"{indent}void unpack(void) override;\n"
        ])

        # TODO: Do we need to check if ndim is 3? In the ideal situation the
        # user will add whatever they need to generate the packet so we probably don't
        # need to check the number of dimensions. 
        # Add extra streams array if necessary.
        if extra_streams > 0: # don't need to release extra queues if we only have 1 stream.
            header.writelines([
                # f"#if MILHOJA_NDIM==3 && defined(MILHOJA_OPENACC_OFFLOADING)\n", # we probably don't need to do this eventually.
                f"{indent}int extraAsynchronousQueue(const unsigned int id) override;\n",
                f"{indent}void releaseExtraQueue(const unsigned int id) override;\n",
                # f"#endif\n",
                f"private:\n",
                f"{indent}static const unsigned int EXTRA_STREAMS = {extra_streams};\n"
                # f"{indent}milhoja::Stream streams_[EXTRA_STREAMS];\n"
            ])
            # TODO: Do this for now until we generate our own task functions
            # header.write("#if MILHOJA_NDIM==3\n")
            for i in range(2, extra_streams+2):
                header.write(f"{indent}milhoja::Stream stream{i}_;\n")
            # header.write("#endif\n")
        else:
            header.writelines([
                "private:\n",
                "\tstd::vector<MemCopyArgs> copyargs;\n"
            ])

        header.writelines([
            f"\tstatic constexpr std::size_t ALIGN_SIZE=16;\n",
            f"\tstatic constexpr std::size_t pad(const std::size_t size) {{ return size + ( ALIGN_SIZE - size % ALIGN_SIZE ) % ALIGN_SIZE; }}\n",
            ''.join(private_variables)
        ])

        level -= 1
        indent = '\t' * level
        header.write("};\n")

        # end start define
        header.write("#endif\n")

    return

# Takes in a file path to load a json file and generates the cpp and header files
def generate_packet_with_filepath(fp, args):
    with open(fp, "r") as file:
        data = json.load(file)
        generate_packet_with_dict(data, args)
        # generate_header_file(data)
        # generate_cpp_file(data)

# gneerate packet data using existing dict
def generate_packet_with_dict(json_dict, args):
    
    if args.cpp:
        generate_cpp_header_file(json_dict)
        generate_cpp_code_file(json_dict)

    if args.fortran:
        generate_fortran_header_file(json_dict)
        generate_fortran_code_file(json_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate packet code files for use in Flash-X problems.")
    parser.add_argument("JSON", help="The JSON file to generate from.")
    parser.add_argument('--cpp', '-c', action="store_true", help="Generate a cpp packet.")
    parser.add_argument("--fortran", '-f', action="store_true", help="Generate a fortran packet.")
    args = parser.parse_args()

    generate_packet_with_filepath(args.JSON, args)