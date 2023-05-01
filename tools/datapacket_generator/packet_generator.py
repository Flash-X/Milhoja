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
import warnings

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

# All possible keys.
EXTRA_STREAMS = 'n-extra-streams'
GENERAL = "general"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_IN_OUT = "tile-in-out"
T_OUT = "tile-out"
# 

# type constants / naming keys / suffixes.
NEW = "new_"
SIZE_T = "std::size_t"
BLOCK_SIZE = "_BLOCK_SIZE_HELPER"
N_TILES = "nTiles"
SCRATCH = "_scratch_d_"
TILE_DESC = "tileDesc_h"
DATA_P = "_data_p_"
DATA_D = "_data_d_"
OUT = "out"
START_P = "_start_p_"
START_D = "_start_d_"
PTRS = "pointers"
PINDEX = f"{PTRS}_index"

SCRATCH_BYTES = "nScratchBytes"
CIN_BYTES = "nCopyInBytes"

HOST = "_h"
PINNED = "_p"
DATA = "_d"
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
constructor_args = []

# TODO: It might be beneficial to write helper methods or a wrapper class for files
# to help with consistently writing to the file, so we
# don't have to put {indent} or \n in every line we write.\

def is_enumerable_type(var):
    return isinstance(var, (dict, list))

def generate_fortran_header_file(parameters, args):
    ...

def generate_fortran_code_file(parameters, args):
    ...

def generate_cpp_code_file(parameters, args):
    def generate_constructor(file, params):
            # function definition
            file.write(f"{params['name']}::{params['name']}({ ', '.join( f'const {item[1]} {NEW}{item[0]}' for item in constructor_args) }) : milhoja::DataPacket{{}}, \n")
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

            for section in params:
                if isinstance(params[section], dict) or isinstance(params[section], list):
                    for item in params[section]:
                        if section == T_MDATA:
                            file.write(f"{indent}{item}{BLOCK_SIZE} = sizeof({mdata.known_types[item]});\n")
                        elif not isinstance(params[section][item], (dict, list)):
                            file.write(f"{indent}{item}{BLOCK_SIZE} = pad(sizeof({params[section][item]}));\n")
                        else:
                            extents, nunkvar, empty = mdata.parse_extents(params[section][item]['extents'], params[section][item]['type'])
                            file.write(f"{indent}{item}{BLOCK_SIZE} = {extents};\n")
            
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
            f"{indent}const Real* data_p = nullptr;\n\n",
            f"{indent}switch (location_) {{\n",
            f"{indent}\tcase PacketDataLocation::CC1: data_p = pinnedPtrs_[n].CC1_data; break;\n",
            f"{indent}\tcase PacketDataLocation::CC2: data_p = pinnedPtrs_[n].CC2_data; break;\n",
            f"{indent}\tdefault: throw std::logic_error(\"[{packet_name}::{func_name}] Data not in CC1 or CC2.\");\n"
            f"{indent}}}\n\n"
        ])

        # data_h and data_p checks
        file.writelines([
            f"{indent}if (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid pointer to data in host memory.\");\n",
            f"{indent}if (data_p == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Invalid pointer to data in pinned memory.\");\n"
        ])

        # TODO: Should we use nunkvars from tile-in?
        file.writelines([
            f"{indent}assert(UNK_VARS_BEGIN == 0);\n"
            f"{indent}assert(UNK_VARS_END == NUNKVAR - 1);\n\n"
        ])

        idx = 0
        last_item = {}
        file.write(f'{indent}{SIZE_T} nBytes;\n')
        for section in [T_IN_OUT, T_OUT]:
            dict = params.get(section, {})
            start_key = 'start' if section == T_OUT else 'start-out'
            end_key = 'end' if section == T_OUT else 'end-out'
            for item in dict:
                start = dict[item][start_key]
                end = dict[item][end_key]
                data_type = dict[item]['type']
                extents, nunkvars, indexer = mdata.parse_extents(dict[item]['extents'])
                num_elems_per_cc_per_var = f'({item}{BLOCK_SIZE} / ( ({nunkvars}) * sizeof({data_type})) )'

                file.writelines([
                    f"{indent}if ( {start} < UNK_VARS_BEGIN || {end} < UNK_VARS_BEGIN || {end} > UNK_VARS_END || {end} - {start} + 1 > {nunkvars})\n",
                    f"{indent}{indent}throw std::logic_error(\"[{packet_name}::{func_name}] Invalid variable mask\");\n\n"
                ])
                
                file.writelines([
                    f"{indent}{SIZE_T} offset_{item} = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>({start});\n",
                ])

                if idx == 0:
                    file.write(f"{indent}{data_type}* start_h = data_h + offset_{item};\n")
                    file.write(f"{indent}const {data_type}* start_p_{item} = data_p + offset_{item};\n")
                else:
                    file.write(f"{indent}start_h += offset_{last_item};\n")
                    file.write(f"{indent}const {data_type}* start_p_{item} = start_p_{last_item} + offset_{item};\n")

                # num_elems_per_cc_per_var = f'(({item}{BLOCK_SIZE}) / ({nunkvars}))'
                file.writelines([
                    f"{indent}nBytes = ({end} - {start} + 1) * ({num_elems_per_cc_per_var}) * sizeof({data_type});\n"
                    f"{indent}std::memcpy((void*)start_h, (void*)start_p_{item}, nBytes);\n\n"                
                ])
                idx += 1
                last_item = item
        
        indent = '\t'
        file.write(f"{indent}}}\n")

        file.write(f"}}\n\n")
        return

    # TODO: Improve pack generation code
    # TODO: {N_TILES} are a guaranteed part of general, same with PacketContents.
    # TODO: We should have constants for variable names in the cpp file generation so they can be easily changed.
    def generate_pack(file, params, args):
        packet_name = params["name"]
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
            f"{indent}Grid& grid = Grid::instance();\n"
        ])

        file.write(f"{indent}{N_TILES} = tiles_.size();\n")
        file.write(f"{indent}{N_TILES}{BLOCK_SIZE} = pad(sizeof(std::size_t));\n\n")

        # SIZE DETERMINATION SECTION
        file.write(f"{indent}/// SIZE DETERMINATION\n")
        # # Scratch section generation.
        file.write(f"{indent}// Scratch section\n")

        bytesToGpu = []
        returnToHost = []
        bytesPerPacket = []

        scratch = params.get(T_SCRATCH, {})
        nScratchArrs = len(scratch)
        size = ' + '.join(f'{item}{BLOCK_SIZE}' for item in scratch) if scratch else "0"
        file.write(f"{indent}{SIZE_T} nScratchPerTileBytes = {size};\n")
        file.write(f"{indent}unsigned int nScratchArrays = {nScratchArrs};\n")
        file.write(f"{indent}{SIZE_T} nScratchPerTileBytesPadded = pad({N_TILES} * nScratchPerTileBytes);\n")
        bytesPerPacket.append("nScratchPerTileBytesPadded")
        file.write(f"{indent}if (nScratchPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] Scratch padding failure\");\n\n")

        # # Copy-in section generation.
        # Non tile specific data
        file.write(f"{indent}// non tile specific data\n")
        file.write(f"{indent}{SIZE_T} nCopyInBytes = nTiles{BLOCK_SIZE} ")

        for item in params.get(GENERAL, []):
            file.write(f"+ {item}{BLOCK_SIZE} ")
        file.write(f"+ {N_TILES} * sizeof(PacketContents);\n") # we can eventually get rid of packet contents so this will have to change.
        file.write(f"{indent}{SIZE_T} nCopyInBytesPadded = pad(nCopyInBytes);\n")
        bytesToGpu.append("nCopyInBytesPadded")
        bytesPerPacket.append("nCopyInBytesPadded")
        file.write(f"{indent}if (nCopyInBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] CopyIn padding failure\");\n\n")

        number_of_arrays = len(params.get(T_IN, {})) + len(params.get(T_IN_OUT, {})) + len(params.get(T_OUT, {}))
        # TODO: If we want to allow any specification for dimensionality of arrays we need to change this
        t_mdata = params.get(T_MDATA, [])
        size = ' + ' + ' + '.join( f'{item}{BLOCK_SIZE}' for item in t_mdata ) if t_mdata else "" 
        file.write(f"{indent}{SIZE_T} nBlockMetadataPerTileBytes = nTiles * ( (nScratchArrays + {number_of_arrays}) * sizeof(FArray4D){size} );\n")
        file.write(f"{indent}{SIZE_T} nBlockMetadataPerTileBytesPadded = pad(nBlockMetadataPerTileBytes);\n")
        bytesToGpu.append("nBlockMetadataPerTileBytesPadded")
        bytesPerPacket.append("nBlockMetadataPerTileBytesPadded")
        file.write(f"{indent}if (nBlockMetadataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] Metadata padding failure\");\n\n")

        # copy in data
        cin = params.get(T_IN, {})
        size = ' + '.join( f'{item}{BLOCK_SIZE}' for item in cin) if cin else "0"
        file.write(f"{indent}{SIZE_T} nCopyInDataPerTileBytes = ({size}) * nTiles;\n")
        file.write(f"{indent}{SIZE_T} nCopyInDataPerTileBytesPadded = pad(nCopyInDataPerTileBytes);\n")
        bytesToGpu.append("nCopyInDataPerTileBytesPadded")
        bytesPerPacket.append("nCopyInDataPerTileBytesPadded")
        file.write(f"{indent}if (nCopyInDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] CopyInPerTile padding failure\");\n\n")

        # copy in out data
        cinout = params.get(T_IN_OUT, {})
        size = ' + '.join( f'{item}{BLOCK_SIZE}' for item in cinout) if cinout else "0"
        file.write(f"{indent}{SIZE_T} nCopyInOutDataPerTileBytes = ({size}) * nTiles;\n")
        file.write(f"{indent}{SIZE_T} nCopyInOutDataPerTileBytesPadded = pad(nCopyInOutDataPerTileBytes);\n")
        bytesToGpu.append("nCopyInOutDataPerTileBytes")
        returnToHost.append("nCopyInOutDataPerTileBytesPadded")
        bytesPerPacket.append("nCopyInOutDataPerTileBytesPadded")
        file.write(f"{indent}if (nCopyInOutDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] CopyInOutPerTile padding failure\");\n\n")

        # copy out
        cout = params.get(T_OUT, {})
        size = ' + '.join( f'{item}{BLOCK_SIZE}' for item in cout ) if cout else "0"
        file.write(f"{indent}{SIZE_T} nCopyOutDataPerTileBytes = ({size}) * nTiles;\n")
        file.write(f"{indent}{SIZE_T} nCopyOutDataPerTileBytesPadded = pad(nCopyOutDataPerTileBytes);\n")
        # bytesToGpu.add("(nTiles * nCopyOutDataPerTileBytes)")
        returnToHost.append("nCopyOutDataPerTileBytes")
        bytesPerPacket.append("nCopyOutDataPerTileBytesPadded")
        file.write(f"{indent}if (nCopyOutDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] CopyOutPerTile padding failure\");\n\n")

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
        file.write(f"{indent}RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - nScratchPerTileBytesPadded, &packet_p_, nBytesPerPacket, &packet_d_);\n")
        file.write(f"{indent}/// END\n\n")
        # END SIZE  DETERMINATION

        if args.sizes:
            s = open(args.sizes, "r")
            sizes = json.load(s)
            s.close()
        else:
            sizes = None

        file.write(f"{indent}/// POINTER DETERMINATION\n")
        file.writelines([
            f"{indent}static_assert(sizeof(char) == 1);\n",
            f"{indent}char* ptr_d = static_cast<char*>(packet_d_);\n\n"
        ])

        file.write(f"{indent}// scratch section\n")
        ### determine scratch pointers
        scr = list( sorted( params.get(T_SCRATCH, {}), key=lambda x: sizes.get(params[T_SCRATCH][x]['type'], 0) if sizes else 1, reverse=True ) )
        for item in scr:
            file.writelines([
                f"{indent}{item}{START_D} = static_cast<void*>(ptr_d);\n",
                f"{indent}ptr_d += nTiles * {item}{BLOCK_SIZE};\n"
            ])
        ###
        file.write(f"{indent}// end scratch\n\n")

        file.writelines([
            f"{indent}location_ = PacketDataLocation::CC1;\n",
            f"{indent}copyInStart_p_ = static_cast<char*>(packet_p_);\n",
            f"{indent}copyInStart_d_ = static_cast<char*>(packet_d_) + nScratchPerTileBytesPadded;\n",
            f"{indent}char* ptr_p = copyInStart_p_;\n"
            f"{indent}ptr_d = copyInStart_d_;\n\n"
        ])

        ### determine general pointers
        general_copy_in_string = ""
        general = sorted(params.get(GENERAL, []), key=lambda x: sizes.get(params[GENERAL][x], 0) if sizes else 1, reverse=True)
        general.insert(0, "nTiles")
        for item in general:
            file.writelines([
                # f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"{indent}{item}{START_P} = static_cast<void*>(ptr_p);\n",
                f"{indent}{item}{START_D} = static_cast<void*>(ptr_d);\n"
                f"{indent}ptr_p += sizeof({item}{BLOCK_SIZE});\n",
                f"{indent}ptr_d += sizeof({item}{BLOCK_SIZE});\n\n"
            ])
            general_copy_in_string += f"{indent}std::memcpy({item}{START_P}, static_cast<void*>(&{item}), {item}{BLOCK_SIZE});\n"
        ###

        ### determine metadata pointers

        ###

        ### determine copy in pointers

        ###

        ### determine copy-in-out pointers

        ### 

        ### determine copy out pointers

        ###
        
        file.writelines([
            # f"{indent}location_ = PacketDataLocation::CC1;\n" # TODO: We need to change this
            # f"{indent}copyInStart_p_ = static_cast<char*>(packet_p_);\n",
            # F"{indent}copyInStart_d_ = static_cast<char*>(packet_d_) + nScratchPerTileBytesPadded;\n"
            f"{indent}copyInOutStart_p_ = copyInStart_p_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded;\n",
            f"{indent}copyInOutStart_d_ = copyInStart_d_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded;\n",
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
        # file.writelines([
        #     f"{indent}char* ptr_p = copyInStart_p_;\n",
        #     f"{indent}ptr_d = copyInStart_d_;\n\n",
        # ])

        # general = sorted(params.get(GENERAL, []), key=lambda x: sizes.get(params[GENERAL][x], 0) if sizes else 1, reverse=True)
        # general.insert(0, "nTiles")
        # for item in general:
        #     file.writelines([
        #         f"//{indent}copyargs[copy_index] = {{ (void*)&{item}, (void*)ptr_p, sizeof({item}{BLOCK_SIZE}) }};\n"
        #         f"//{indent}copy_index++;\n"
        #         f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
        #         f"{indent}ptr_p += sizeof({item}{BLOCK_SIZE});\n",
        #         f"{indent}ptr_d += sizeof({item}{BLOCK_SIZE});\n\n"
        #     ])

        # packet contents comes after general section in hydro variants.
        file.writelines([
            f"{indent}contents_p_ = static_cast<PacketContents*>((void*)ptr_p);\n",
            f"{indent}contents_d_ = static_cast<PacketContents*>((void*)ptr_d);\n",
            f"{indent}ptr_p += {N_TILES} * sizeof(PacketContents);\n",
            f"{indent}ptr_d += {N_TILES} * sizeof(PacketContents);\n\n"
        ])

        previous = ""
        data_copy_string = ""
        for idx,item in enumerate( sorted( params.get(T_IN, {}), key=lambda x: sizes.get(params[T_IN][x]['type'], 0) if sizes else 1, reverse=True ) ):
            start = params[T_IN][item]['start']
            end = params[T_IN][item]['end']
            data_type = params[T_IN][item]['type']
            extents, nunkvars, empty = mdata.parse_extents(params[T_IN][item]['extents'])
            num_elems_per_cc_per_var = f'({item}{BLOCK_SIZE} / ( ({nunkvars}) * sizeof({data_type})) )'
            offset = f"{indent*2}{SIZE_T} offset_{item} = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>({start});\n"
            copy_in_size = f"{indent*2}{SIZE_T} nBytes_{item} = ({end} - {start} + 1) * ({num_elems_per_cc_per_var}) * sizeof({data_type});\n"
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyInStart_p_ + nCopyInBytes + nBlockMetadataPerTileBytesPadded;\n")
                file.write(f"{indent}char* {item}{START_D} = copyInStart_d_ + nCopyInBytes + nBlockMetadataPerTileBytesPadded;\n")
                data_copy_string += offset
                data_copy_string += copy_in_size
                data_copy_string += f"{indent*2}std::memcpy((void*){item}{START_P}, (void*)(data_h + offset_{item}), nBytes_{item});\n"
                if previous == "":
                    previous = f"data_h + {item}{BLOCK_SIZE} "
                else:
                    previous += f"+ {item}{BLOCK_SIZE}"
            else:
                l = list(params[T_IN])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {l[idx-1]}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_D} = {l[idx-1]}{START_D} + {l[idx-1]}{BLOCK_SIZE};\n")
                data_copy_string += offset
                data_copy_string += copy_in_size
                data_copy_string += f"{indent*2}std::memcpy((void*){item}{START_P}, (void*)({previous} + offset_{item}), nBytes_{item});\n"
                previous += f"+ {item}{BLOCK_SIZE}"

        # TODO: We don't need to worry about data location since generated code should automatically know the data location.
        # TODO: When do we change where the start of CC1 and CC2 data is located?
        # for item in params.get(T_IN_OUT, {}):
        # TODO: We need to change this, T_OUT, and T_IN to work like the scratch section
        for idx,item in enumerate( sorted( params.get(T_IN_OUT, {}), key=lambda x: sizes.get(params[T_IN_OUT][x]['type'], 0) if sizes else 1, reverse=True ) ):
            start = params[T_IN_OUT][item]['start-in']
            end = params[T_IN_OUT][item]['end-in']
            data_type = params[T_IN_OUT][item]['type']
            extents, nunkvars, empty = mdata.parse_extents(params[T_IN_OUT][item]['extents'])
            num_elems_per_cc_per_var = f'({item}{BLOCK_SIZE} / ( ({nunkvars}) * sizeof({data_type})) )'
            offset = f"{indent*2}{SIZE_T} offset_{item} = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>({start});\n"
            copy_in_size = f"{indent*2}{SIZE_T} nBytes_{item} = ({end} - {start} + 1) * ({num_elems_per_cc_per_var}) * sizeof({data_type});\n"
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyInOutStart_p_;\n")#+ nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
                file.write(f"{indent}char* {item}{START_D} = copyInOutStart_d_;\n")# + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
                data_h = f"data_h + offset_{item}" if previous == "" else previous
                data_copy_string += offset
                data_copy_string += copy_in_size
                data_copy_string += f"{indent*2}std::memcpy((void*){item}{START_P}, (void*)({data_h}), nBytes_{item});\n"
                if previous == "":
                    previous = f"data_h + {item}{BLOCK_SIZE} "
                else:
                    previous += f"+ {item}{BLOCK_SIZE}"
            else:
                l = list(params[T_IN_OUT])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {l[idx-1]}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_D} = {l[idx-1]}{START_D} + {l[idx-1]}{BLOCK_SIZE};\n")
                data_copy_string += offset
                data_copy_string += copy_in_size
                data_copy_string += f"{indent*2}std::memcpy((void*){item}{START_P}, (void*)({previous} + offset_{item}), nBytes_{item});\n"
                previous += f" + {item}{BLOCK_SIZE}"
         
        for idx,item in enumerate( sorted( params.get(T_OUT, {}), key=lambda x: sizes.get(params[T_OUT][x]['type'], 0) if sizes else 1, reverse=True ) ):
            if idx == 0:
                file.write(f"{indent}char* {item}{START_P} = copyOutStart_p;\n")# + {N_TILES} * copyInOutDataPerTileBytes;\n")
                file.write(f"{indent}char* {item}{START_D} = copyOutStart_d;\n")# + {N_TILES} * copyInOutDataPerTileBytes;\n")
            else:
                l = list(params[T_OUT])
                file.write(f"{indent}char* {item}{START_P} = {l[idx-1]}{START_P} + {l[idx-1]}{BLOCK_SIZE};\n")
                file.write(f"{indent}char* {item}{START_D} = {l[idx-1]}{START_D} + {l[idx-1]}{BLOCK_SIZE};\n")

        # Create all scratch ptrs.
        # scr = sorted(list(params.get(T_SCRATCH, {}).keys()))
        # scr = list( sorted( params.get(T_SCRATCH, {}), key=lambda x: sizes.get(params[T_SCRATCH][x]['type'], 0) if sizes else 1, reverse=True ) )
        # for idx,item in enumerate(scr):
        #     if idx == 0:  # we can probably use an iterator here instead
        #         file.write(f"{indent}char* {scr[idx]}{START_D} = scratchStart_d;\n")
        #     else:
        #         file.write(f"{indent}char* {scr[idx]}{START_D} = {scr[idx-1]}{START_D} + {scr[idx-1]}{BLOCK_SIZE};\n")
        file.write(f"{indent}PacketContents* tilePtrs_p = contents_p_;\n")
        file.write(f"{indent}char* char_ptr;\n")

        file.write(general_copy_in_string + "\n")
            
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

        # Memcpy copy-in/copy-in-out data with data_h as source
        file.write(data_copy_string)

        # be careful here, is pinnedptrs tied to tile-in-out or tile-out? What about tile-in?
        # We need to change this so that we aren't accidentally assigning CC1_data to a cc2 ptr.
        # TODO: Revisit when packet contents is removed
        if T_IN_OUT in params:
            nxt = next(iter(params[T_IN_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = static_cast<{params[T_IN_OUT][nxt]['type']}*>((void*){ nxt }{START_P});\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = nullptr;\n")

        if T_OUT in params:
            nxt = next(iter(params[T_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = static_cast<{params[T_OUT][nxt]['type']}*>((void*){ nxt }{START_P});\n\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = nullptr;\n\n")

        # TODO: Could we possibly merge T_MDATA and the device_array_pointers sections?
        # There's the obvious way of just using an if statement based on the section... but is there a better way?
        # Add metadata to ptr
        for item in sorted(params.get(T_MDATA, []), key=lambda x: sizes.get(mdata.known_types[x], 0) if sizes else 1, reverse=True):
            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<{mdata.known_types[item]}*>((void*)ptr_d);\n",
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"//{indent}copyargs[copy_index] = {{ (void*)&{item}, (void*)ptr_p, {item}{BLOCK_SIZE} }};\n",
                f"//{indent}copy_index++;\n",
                f"{indent}ptr_p += {item}{BLOCK_SIZE};\n"
                f"{indent}ptr_d += {item}{BLOCK_SIZE};\n\n"
            ])

        for item in sorted(device_array_pointers, key=lambda x: sizes[device_array_pointers[x]['type']] if sizes else 1, reverse=True ):
            d = 4 # assume d = 4 for now.
            section = device_array_pointers[item]['section']
            type = device_array_pointers[item]['type']
            extents, nunkvars, indexer = mdata.parse_extents(device_array_pointers[item]['extents'])
            c_args = mdata.constructor_args[indexer]

            file.write(f"{indent}tilePtrs_p->{item}_d = static_cast<FArray{d}D*>((void*)ptr_d);\n")
            file.writelines([
                f"{indent}FArray{d}D {item}_d{{ static_cast<{type}*>((void*){item}{START_D}), {c_args}, {nunkvars}}};\n"
                f"//{indent}copyargs[copy_index] = {{ (void*)&{item}_d, (void*)ptr_p, sizeof(FArray{d}D) }};\n",
                f"//{indent}copy_index++;\n",
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}_d, sizeof(FArray{d}D));\n",
                f"{indent}ptr_p += sizeof(FArray{d}D);\n",
                f"{indent}ptr_d += sizeof(FArray{d}D);\n",
            ])

            if section == T_SCRATCH:
                file.write(f"{indent}{item}{START_D} = static_cast<void*>( static_cast<char*>({item}{START_D}) + {item}{BLOCK_SIZE} );\n\n")
            elif section == T_IN or section == T_IN_OUT:
                file.write(f"{indent}{item}{START_P} += nBytes_{item};\n")
                file.write(f"{indent}{item}{START_D} += nBytes_{item};\n\n")
            else:
                file.write(f"{indent}{item}{START_P} += {item}{BLOCK_SIZE};\n")
                file.write(f"{indent}{item}{START_D} += {item}{BLOCK_SIZE};\n\n")

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
            f"{indent}if (!stream_.isValid()) throw std::runtime_error(\"[{packet_name}::pack] Unable to acquire stream\");\n",
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
        
        file.write(f"}}\n\n")
        return
    
    # Generate clone method
    def generate_clone(file, params):
        packet_name = params["name"]
        file.writelines([
            f"std::unique_ptr<milhoja::DataPacket> {packet_name}::clone(void) const {{\n",
            f"\treturn std::unique_ptr<milhoja::DataPacket>{{ new {packet_name}{{{', '.join( f'{item[0]}' for item in constructor_args)}}} }};\n"
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
        generate_pack(code, parameters, args)
        generate_unpack(code, parameters)

# Creates a header file based on the given parameters.
# Lots of boilerplate and class member generation.
# TODO: Pack and unpack functions use a specific variable involving CC1 and CC2. How can we specify this in tile-in and tile-out?
def generate_cpp_header_file(parameters, args):
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
        pinned_and_data_ptrs = ""
        
        # define statements
        header.write(f"#ifndef {defined}\n")
        header.write(f"#define {defined}\n")
        # boilerplate includes
        header.write("#include <Milhoja.h>\n")
        header.write("#include <Milhoja_DataPacket.h>\n")

        # manually generate nTiles getter here
        pinned_and_data_ptrs += f"\t{SIZE_T} nTiles;\n\tvoid* nTiles{START_P};\n\tvoid* nTiles{START_D};\n"
        private_variables.append(f"\t{SIZE_T} nTiles{BLOCK_SIZE};\n")

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
                item_type = general[item] if not is_enumerable else general[item]['type']
                if is_enumerable: types.add( f"FArray4D" )
                # private_variables.append(var)
                private_variables.append(size_var)
                vars_and_types[block_size_var] = f"milhoja::{general[item]}"
                types.add(item_type)
                if item_type in mdata.imap: item_type = f"milhoja::{item_type}"
                constructor_args.append([item, item_type])
                # be careful here we can't assume all items in general are based in milhoja
                pinned_and_data_ptrs += "\t"
                if type in mdata.imap: 
                    pinned_and_data_ptrs += "milhoja::"
                pinned_and_data_ptrs += f"void* {item}{START_P};\n\tvoid* {item}{START_D};\n"

        # Generate private variables for each section. Here we are creating a size helper
        # variable for each item in each section based on the name of the item
        if T_MDATA in parameters:
            for item in parameters[T_MDATA]:
                new_variable = f"{item}{BLOCK_SIZE}"
                if mdata.known_types[item]:
                    types.add( mdata.known_types[item] )
                else:
                    print("Found bad data in tile-metadata. Ignoring...")
                    continue
                private_variables.append(f"\t{SIZE_T} {new_variable};\n")
                vars_and_types[new_variable] = SIZE_T
                pinned_and_data_ptrs += f"\tvoid* {item}{START_P};\n\tvoid* {item}{START_D};\n"

        for sect in [T_IN, T_IN_OUT, T_OUT, T_SCRATCH]:
            for item in parameters.get(sect, {}):
                private_variables.append(f"\t{SIZE_T} {item}{BLOCK_SIZE};\n")
                is_enumerable = is_enumerable_type(parameters[sect][item])
                types.add(parameters[sect][item] if not is_enumerable_type(parameters[sect][item]) else parameters[sect][item]['type'])
                if is_enumerable: types.add( f"FArray4D" )
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T
                device_array_pointers[item] = {"section": sect, **parameters[sect][item]}

                if sect != T_SCRATCH:
                    pinned_and_data_ptrs += f"\tvoid* {item}{START_P};\n"
                pinned_and_data_ptrs += f"\tvoid* {item}{START_D};\n"


        # we only want to include things if they are found in the include dict.
        header.write( ''.join( f"#include {mdata.imap[item]}\t\n" for item in types if item in mdata.imap) )

        # header.writelines([
        #     f"\nstruct MemCopyArgs {{\n",
        #     f"\tvoid* src = nullptr;\n",
        #     f"\tvoid* dest = nullptr;\n",
        #     f"\t{SIZE_T} size = 0;\n"
        #     f"}};\n\n"
        # ])

        # class definition
        header.write(f"class {name} : public milhoja::DataPacket {{ \n")
        level += 1
        indent = '\t' * level

        # public information
        header.write("public:\n")
        header.write(f"{indent}std::unique_ptr<milhoja::DataPacket> clone(void) const override;\n")
        header.write(f"{indent}{name}({', '.join(f'const {item[1]} {NEW}{item[0]}' for item in constructor_args)});\n")
        # header.write(indent + f"{name}(const milhoja::Real {NEW}dt);\n")
        header.write(indent + f"~{name}(void);\n")

        # Constructors & = operations
        header.writelines([
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
                #"\tMemCopyArgs* copyargs;\n"
            ])

        header.writelines([
            f"\tstatic constexpr std::size_t ALIGN_SIZE={parameters.get('byte-align', 16)};\n",
            f"\tstatic constexpr std::size_t pad(const std::size_t size) {{ return ((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE; }}\n",
            ''.join( f'\t{item[1]} {item[0]};\n' for item in constructor_args),
            ''.join(private_variables),
            ''.join(pinned_and_data_ptrs)
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
        generate_cpp_header_file(json_dict, args)
        generate_cpp_code_file(json_dict, args)

    if args.fortran:
        generate_fortran_header_file(json_dict, args)
        generate_fortran_code_file(json_dict, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate packet code files for use in Flash-X problems.")
    parser.add_argument("JSON", help="The JSON file to generate from.")
    parser.add_argument('--cpp', '-c', action="store_true", help="Generate a cpp packet.")
    parser.add_argument("--fortran", '-f', action="store_true", help="Generate a fortran packet.")
    parser.add_argument("--sizes", "-s", help="Path to data type size information.")
    args = parser.parse_args()

    if args.sizes: print(args.sizes)
    else: print("No sizes path found...")

    generate_packet_with_filepath(args.JSON, args)