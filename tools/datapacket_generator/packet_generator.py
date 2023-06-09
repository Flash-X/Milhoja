#!/usr/bin/env python
"""
Author: Wesley Kwiecinski

Packet generator for Milhoja. Script takes in a 
JSON file and generates cpp code for a data packet
based on the contents of the json file.

"""

import os.path
import json
import argparse
import milhoja_utility as mdata
import json_sections
import warnings
from typing import TextIO

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

HOST = "_h_"
PINNED = "_p_"
DATA = "_d_"

# All possible keys.
GENERAL = json_sections.GENERAL
T_SCRATCH = json_sections.T_SCRATCH
T_MDATA = json_sections.T_MDATA
T_IN = json_sections.T_IN
T_IN_OUT = json_sections.T_IN_OUT
T_OUT = json_sections.T_OUT

EXPANDED = json_sections.EXPANDED
NUNK = json_sections.NUNK
INDEXER = json_sections.INDEXER
NUM_ELEMS = json_sections.NUM_ELEMS

EXTRA_STREAMS = 'n-extra-streams'
START = "start"
END = "end"
START_IN = "start-in"
END_IN = "end-in"
# 

# type constants / naming keys / suffixes.
NEW = "new_"
SIZE_T = "std::size_t"
BLOCK_SIZE = "_BLOCK_SIZE_HELPER"
N_TILES = f"nTiles{HOST}"
SCRATCH = "_scratch_d_"
TILE_DESC = "tileDesc_h"
DATA_P = "_data_p_"
DATA_D = "_data_d_"
OUT = "out"
START_P = "_start_p_"
START_D = "_start_d_"
PTRS = "pointers"
PINDEX = f"{PTRS}_index"
GETTER = "_devptr"

SCRATCH_BYTES = "nScratchBytes"
# 
# these might not be necessary
initialize = {}

# TODO: Maybe we can create dictionaries derived from sections in the json? Something to think about
device_array_pointers = {}

all_pointers = set()
includes = set()
constructor_args = []
farray_items = []
location = ""

def is_enumerable_type(var):
    """Returns True if var is a dict or list."""
    return isinstance(var, (dict, list))

def generate_cpp_code_file(parameters: dict, args):
    """
    Generates the .cpp file for the data packet.

    Parameters
        parameters - The json file parameters.\n
        args - The namespace containing the arguments passed in from the command line
    Returns:
        None
    """
    # TODO: Move all size gereration to constructor or at beginning of pack function.
    #       They should not be split up
    def generate_constructor(file: TextIO, params: dict):
        """
        Generates the constructor function for the data packet based on the JSON file.

        Parameters
            file -   The file to write to.\n
            params - The JSON dictionary.
        Returns:
            None
        """
        indent = "\t"
        # function definition
        file.writelines([
            f"{params['name']}::{params['name']}({ ', '.join( f'{item[1]} {NEW}{item[0]}' for item in constructor_args) }) : milhoja::DataPacket{{}}, \n",
            f"".join(f"{indent}{variable}{{ {initialize[variable] } }},\n" for variable in initialize),
            f",\n".join(f"{indent}{item}{HOST}{{new_{item}}}" for item in params.get(GENERAL, [])),
            "\n{\n",
            f"{indent}using namespace milhoja;\n",
            f"{indent}milhoja::Grid::instance().getBlockSize(&nxb_, &nyb_, &nzb_);\n",
        ])

        file.write("}\n\n")

    # TODO: Eventually come back and fix the streams to use the array implementation.
    def generate_destructor(file: TextIO, params: dict):
        """
        Generates the destructor for the data packet.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary
        Returns:
            None
        """
        p_name = params["name"]
        extra_streams = params.get(EXTRA_STREAMS, 0)
        file.write( f"{p_name}::~{p_name}(void) {{\n" )
        file.writelines([ f"\tif (stream{i}_.isValid()) throw std::logic_error(\"[{p_name}::~{p_name}] Stream {i} not released\");\n" for i in range(2, extra_streams+2) ])
        file.write( f"\tnullify();\n}}\n\n" ) 

    # TODO: Some parts of the unpack method need to be more generalized.
    def generate_unpack(file: TextIO, params: dict):
        """
        Generates the unpack function for the data packet.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary
        Returns:
            None
        """
        packet_name = params["name"]
        func_name = "unpack"
        # num_elems_per_cc_per_var = 'ELEMS_PER_CC_PER_VAR'
        nunkvars = "nCcVars_"
        
        file.writelines([
            f"void {packet_name}::unpack(void) {{\n",
            "\tusing namespace milhoja;\n",
            f"\tif (tiles_.size() <= 0) throw std::logic_error(\"[{packet_name}::{func_name}] Empty data packet.\");\n",
            f"\tif (!stream_.isValid()) throw std::logic_error(\"[{packet_name}::{func_name}] Stream not acquired.\");\n",
            f"\tif (pinnedPtrs_ == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] No pinned pointers set.\");\n"
            "\tRuntimeBackend::instance().releaseStream(stream_);\n"
            "\tassert(!stream_.isValid());\n",
            f"\tint   nxbGC_h     = -1;\n",
            f"\tint   nybGC_h     = -1;\n",
            f"\tint   nzbGC_h     = -1;\n",
            f"\tint   nCcVars_h   = -1;\n",
            f"\tint   nFluxVars_h = -1;\n",
            f"\ttileSize_host(&nxbGC_h, &nybGC_h, &nzbGC_h, &nCcVars_h, &nFluxVars_h);\n\n"
        ])

        file.writelines([
            "\tfor (auto n=0; n < tiles_.size(); ++n) {\n",
            "\t\tTile* tileDesc_h = tiles_[n].get();\n",
            "\t\tReal* data_h = tileDesc_h->dataPtr();\n",
            f"\t\tconst Real* data_p = pinnedPtrs_[n].{ 'CC1' if T_IN_OUT in params else 'CC2' }_data;\n", # TODO: This doesn't work with the location key in JSON
            f"\t\tif (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid pointer to data in host memory.\");\n",
            f"\t\tif (data_p == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Invalid pointer to data in pinned memory.\");\n",
            f"\t\t{SIZE_T} nBytes;\n\n"
        ])

        output = {**params.get(T_IN_OUT, {}), **params.get(T_OUT, {}) }
        for item in output:
            start_key = START if START in output[item] else f"{START}-out"
            end_key = END if END in output[item] else f"{END}-out"
            start = output[item][start_key]
            end = output[item][end_key]
            data_type = output[item]['type']
            num_elems_per_cc_per_var = ' * '.join(output[item][NUM_ELEMS])
            file.writelines([
                f"\t\tif ( {start} < 0 || {end} < 0 || {end} >= {nunkvars} || {start} >= {nunkvars})\n",
                f"\t\t\tthrow std::logic_error(\"[{packet_name}::{func_name}] Invalid variable mask\");\n"
                f"\t\tif ( {start} > {end} )\n",
                f"\t\t\tthrow std::logic_error(\"{packet_name}::{func_name}] Start > End\");\n\n",
                f"\t\t{SIZE_T} offset_{item} = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>({start});\n",
                f"\t\t{data_type}* start_h = data_h + offset_{item};\n",
                f"\t\tconst {data_type}* start_p_{item} = data_p + offset_{item};\n",
                f"\t\tnBytes = ( ({end}) - ({start}) + 1 ) * ({num_elems_per_cc_per_var}) * sizeof({data_type});\n",
                f"\t\tstd::memcpy(static_cast<void*>(start_h), static_cast<const void*>(start_p_{item}), nBytes);\n\n"
            ])
        file.writelines([
            "\t}\n",
            "}\n\n"
        ])

    # TODO: Improve pack generation code
    # TODO: {N_TILES} are a guaranteed part of general, same with PacketContents.
    def generate_pack(file: TextIO, params: dict, args):
        """
        Generate the pack function for the data packet.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary\n
            args - The namespace containing the arguments from the command line
        Returns:
            None
        """
        packet_name = params["name"]
        func_name = "pack"
        indent = "\t"
        
        # Error checking
        file.writelines([
            f"void {packet_name}::pack(void) {{\n",
            f"{indent}using namespace milhoja;\n",
            f"{indent}std::string errMsg = isNull();\n",
            f"{indent}if (errMsg != \"\")\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent}else if (tiles_.size() == 0)\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] No tiles added.\");\n"
            f"{indent}\n",
            f"{indent}/// SIZE DETERMINATION\n"
        ])

        def write_block_sizes() -> None:
            """Writes the block sizes of each item in the JSON."""
            # some misc constructor code for calculating block sizes.
            if args.language == mdata.Language.fortran:
                file.writelines([
                    f"\tint   nxbGC_h     = -1;\n",
                    f"\tint   nybGC_h     = -1;\n",
                    f"\tint   nzbGC_h     = -1;\n",
                    f"\tint   nCcVars_h   = -1;\n",
                    f"\tint   nFluxVars_h = -1;\n",
                    f"\ttileSize_host(&nxbGC_h, &nybGC_h, &nzbGC_h, &nCcVars_h, &nFluxVars_h);\n"
                ])    
            def generate_size_string(subitem_dict: dict, args) -> str:
                """Generates the size string for a non metadata item in the JSON."""
                if isinstance(subitem_dict, str): 
                    return f"pad(sizeof({subitem_dict}))"
                return subitem_dict[EXPANDED]

            def generate_mdata_size(mdata_item):
                """Generates the size string for a metadata item in the JSON"""
                if args.language != mdata.Language.cpp and mdata.tile_known_types[mdata_item] in mdata.cpp_equiv:
                    return f"MILHOJA_MDIM * sizeof({mdata.cpp_equiv[mdata.tile_known_types[mdata_item]]})"
                return f"sizeof({mdata.tile_known_types[mdata_item]})"

            sections = [ T_SCRATCH, GENERAL, T_MDATA, T_IN, T_IN_OUT, T_OUT ]
            for item in sections:
                item = params.get(item)
                if isinstance(item, list): # mdata section
                    file.writelines([ f"\t{SIZE_T} {subitem}{BLOCK_SIZE} = {generate_mdata_size(subitem)};\n" for subitem in item ])
                elif isinstance(item, dict): # other dict
                    file.writelines([ f"\t{SIZE_T} {subitem}{BLOCK_SIZE} = {generate_size_string(item[subitem], args)};\n" for subitem in item ])  
        # SIZE DETERMINATION SECTION

        write_block_sizes()

        file.writelines([
            f"{indent}{N_TILES} = tiles_.size();\n"
            f"{indent}{SIZE_T} nTiles{BLOCK_SIZE} = sizeof(int);\n"
        ])

        # # Scratch section generation.
        file.write(f"\n{indent}// Scratch section\n")
        bytesToGpu = []
        returnToHost = []
        bytesPerPacket = []
        scratch = params.get(T_SCRATCH, {})
        nScratchArrs = len(scratch)

        # # Copy-in section generation.
        # Non tile specific data
        file.writelines([
            f"{indent}// non tile specific data\n",
            f"{indent}{SIZE_T} generalDataBytes = nTiles{BLOCK_SIZE} "
        ])
        p_contents_size = f" + {N_TILES} * sizeof(PacketContents)" if args.language == mdata.Language.cpp else ""
        if args.language == mdata.Language.fortran:
            p_contents_size = ""
        file.writelines([ 
            f" + { ' + '.join(f'{item}{BLOCK_SIZE}' for item in params.get(GENERAL, [])) }",
            f"{p_contents_size};\n", # we can eventually get rid of packet contents so this will have to change.
            f"{indent}{SIZE_T} generalDataBytesPadded = pad(generalDataBytes);\n",
            f"{indent}if (generalDataBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] General padding failure\");\n\n"
        ])
        bytesToGpu.append("generalDataBytesPadded")
        bytesPerPacket.append("generalDataBytesPadded")

        number_of_arrays = len(params.get(T_IN, {})) + len(params.get(T_IN_OUT, {})) + len(params.get(T_OUT, {}))
        # TODO: If we want to allow any specification for dimensionality of arrays we need to change this
        t_mdata = params.get(T_MDATA, [])
        size = ' + ' + ' + '.join( f'{item}{BLOCK_SIZE}' for item in t_mdata ) if t_mdata else "" 
        scratch_arrays = "0" if args.language != mdata.Language.cpp else f"( ({nScratchArrs} + {number_of_arrays}) * sizeof(FArray4D) )"
        file.writelines([
            f"{indent}{SIZE_T} nBlockMetadataPerTileBytes = {N_TILES} * ( {scratch_arrays}{size} );\n",
            f"{indent}{SIZE_T} nBlockMetadataPerTileBytesPadded = pad(nBlockMetadataPerTileBytes);\n",
            f"{indent}if (nBlockMetadataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] Metadata padding failure\");\n\n"
        ])
        bytesToGpu.append("nBlockMetadataPerTileBytesPadded")
        bytesPerPacket.append("nBlockMetadataPerTileBytesPadded")

        array_sections = [T_IN, T_IN_OUT, T_OUT, T_SCRATCH]
        for section in array_sections:
            sect_dict = params.get(section, {})
            size = ' + '.join( f'{item}{BLOCK_SIZE}' for item in sect_dict) if sect_dict else "0"
            sect = section.replace('-', '')
            file.writelines([
                f"{indent}{SIZE_T} {sect}DataBytes = ({size}) * {N_TILES};\n",
                f"{indent}{SIZE_T} {sect}DataBytesPadded = pad({sect}DataBytes);\n",
                f"{indent}if ({sect}DataBytesPadded % ALIGN_SIZE != 0) throw std::logic_error(\"[{packet_name}] {sect}Bytes padding failure\");\n\n"
            ])
            if 'in' in sect:
                bytesToGpu.append(f"{sect}DataBytesPadded")
            if 'out' in sect:
                returnToHost.append(f"{sect}DataBytesPadded")
            bytesPerPacket.append(f"{sect}DataBytesPadded")

        file.writelines([
            f"{indent}// Copy out section\n",
            f"{indent}nCopyToGpuBytes_ = {' + '.join(bytesToGpu)};\n",
            f"{indent}nReturnToHostBytes_ = {' + '.join(returnToHost)};\n",
            f"{indent}{SIZE_T} nBytesPerPacket = {' + '.join(bytesPerPacket)};\n",
            f"{indent}RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - {T_SCRATCH.replace('-', '')}DataBytesPadded, &packet_p_, nBytesPerPacket, &packet_d_);\n",
            f"{indent}/// END\n\n"
        ])

        # END SIZE  DETERMINATION

        sizes = None
        if args.sizes:
            with open(args.sizes, "r") as s:
                sizes = json.load(s)

        file.writelines([
            f"{indent}/// POINTER DETERMINATION\n",
            f"{indent}static_assert(sizeof(char) == 1);\n",
            f"{indent}char* ptr_d = static_cast<char*>(packet_d_);\n\n",
            f"{indent}// scratch section\n"
        ])

        ### DETERMINE SCRATCH POINTERS
        scr = sorted( params.get(T_SCRATCH, {}), key=lambda x: sizes.get(params[T_SCRATCH][x]['type'], 0) if sizes else 1, reverse=True )
        file.writelines([ f"{indent}{item}{START_D} = static_cast<void*>(ptr_d);\n{indent}ptr_d += {N_TILES} * {item}{BLOCK_SIZE};\n" for item in scr ])
        file.write(f"{indent}// end scratch\n\n")
        ###

        location = "CC1"
        file.writelines([
            f"{indent}location_ = PacketDataLocation::{location};\n",
            f"{indent}copyInStart_p_ = static_cast<char*>(packet_p_);\n",
            f"{indent}copyInStart_d_ = static_cast<char*>(packet_d_) + {T_SCRATCH.replace('-', '')}DataBytesPadded;\n",
            f"{indent}char* ptr_p = copyInStart_p_;\n"
            f"{indent}ptr_d = copyInStart_d_;\n\n",
            f"\t// general section;\n"
        ])

        ### DETERMINE GENERAL POINTERS
        general_copy_in = []
        params.get(GENERAL, {})["nTiles"] = f"{SIZE_T}"
        general = sorted(params.get(GENERAL, []), key=lambda x: sizes.get(params[GENERAL][x], 0) if sizes else 1, reverse=True)
        # Note: We add nTiles to general to make generation easier but nTiles cannot be const because it is set in pack().
        # I could probably set nTiles in the constructor... I'd have to make sure that setting it in the constructor works fine.
        for item in general:
            file.writelines([
                f"{indent}{item}{START_P} = static_cast<void*>(ptr_p);\n",
                f"{indent}{item}{START_D} = static_cast<void*>(ptr_d);\n"
                f"{indent}ptr_p += sizeof({item}{BLOCK_SIZE});\n",
                f"{indent}ptr_d += sizeof({item}{BLOCK_SIZE});\n\n"
            ])
            const = "const " if item != N_TILES else ""
            general_copy_in.append(f"{indent}std::memcpy({item}{START_P}, static_cast<{const}void*>(&{item}{HOST}), {item}{BLOCK_SIZE});\n")
        if args.language == mdata.Language.cpp:
            file.writelines([
                f"{indent}contents_p_ = static_cast<PacketContents*>( static_cast<void*>(ptr_p) );\n",
                f"{indent}contents_d_ = static_cast<PacketContents*>( static_cast<void*>(ptr_d) );\n",
                f"{indent}ptr_p += {N_TILES} * sizeof(PacketContents);\n",
                f"{indent}ptr_d += {N_TILES} * sizeof(PacketContents);\n"
            ])
        file.write("\t// end general\n\n")
        ###

        ### DETERMINE METADATA POINTERS
        file.write("\t// metadata section;\n")
        metadata = sorted(params.get(T_MDATA, []), key=lambda x: sizes.get(mdata.tile_known_types[x], 0) if sizes else 1, reverse=True)
        for item in metadata:
            file.writelines([
                f"{indent}{item}{START_P} = static_cast<void*>(ptr_p);\n",
                f"{indent}{item}{START_D} = static_cast<void*>(ptr_d);\n",
                f"{indent}ptr_p += {N_TILES} * {item}{BLOCK_SIZE};\n",
                f"{indent}ptr_d += {N_TILES} * {item}{BLOCK_SIZE};\n\n"
            ])
        if args.language == mdata.Language.cpp:
            for item in sorted(farray_items):
                file.writelines([
                    f"{indent}char* {item}_farray_start_p_ = ptr_p;\n",
                    f"{indent}char* {item}_farray_start_d_ = ptr_d;\n"
                    f"{indent}ptr_p += {N_TILES} * sizeof(FArray4D);\n",
                    f"{indent}ptr_d += {N_TILES} * sizeof(FArray4D);\n\n"
                ])
        ###

        ### DETERMINE COPY IN POINTERS
        file.writelines([
            "\t// end metadata;\n\n",
            "\t// copy in section;\n",
            f"{indent}ptr_p = copyInStart_p_ + generalDataBytesPadded + nBlockMetadataPerTileBytesPadded;\n",
            f"{indent}ptr_d = copyInStart_d_ + generalDataBytesPadded + nBlockMetadataPerTileBytesPadded;\n\n"
        ])

        def write_section_pointers(section, item_key, start_key, end_key, copy_string, fp):
            start = params[section][item_key][start_key]
            end = params[section][item_key][end_key]
            data_type = params[section][item_key]['type']
            nunkvars = params[section][item_key][NUNK]
            num_elems_per_cc_per_var = ' * '.join( params[section][item_key][NUM_ELEMS] )
            fp.writelines([
                f"\t{item_key}{START_P} = static_cast<void*>(ptr_p);\n",
                f"\t{item_key}{START_D} = static_cast<void*>(ptr_d);\n",
                f"\tptr_p += {N_TILES} * {item_key}{BLOCK_SIZE};\n",
                f"\tptr_d += {N_TILES} * {item_key}{BLOCK_SIZE};\n\n"
            ])
            copy_string.extend([
                f"\t\t{SIZE_T} offset_{item_key} = ({num_elems_per_cc_per_var}) * static_cast<{SIZE_T}>({start});\n",
                f"\t\t{SIZE_T} nBytes_{item_key} = ( ({end}) - ({start}) + 1 ) * ({num_elems_per_cc_per_var}) * sizeof({data_type});\n",
                f"\t\tchar_ptr = static_cast<char*>({item_key}{START_P}) + n * {item_key}{BLOCK_SIZE};\n",
                f"\t\tstd::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(data_h + offset_{item_key}), nBytes_{item_key});\n"
            ])

        # TILE-IN POINTERS
        data_copy = []
        for item in sorted( params.get(T_IN, {}), key=lambda x: sizes.get(params[T_IN][x]['type'], 0) if sizes else 1, reverse=True ):
            write_section_pointers( T_IN, item, 'start', 'end', data_copy, file )
        ###

        file.writelines([
            "\t// end copy in;\n\n",
            f"\t// copy in out section\n",
            f"{indent}copyInOutStart_p_ = copyInStart_p_ + generalDataBytesPadded + nBlockMetadataPerTileBytesPadded + {T_IN.replace('-', '')}DataBytesPadded;\n",
            f"{indent}copyInOutStart_d_ = copyInStart_d_ + generalDataBytesPadded + nBlockMetadataPerTileBytesPadded + {T_IN.replace('-', '')}DataBytesPadded;\n",
            f"{indent}ptr_p = copyInOutStart_p_;\n"
            f"{indent}ptr_d = copyInOutStart_d_;\n\n"
        ])

        ### DETERMINE COPY-IN-OUT POINTERS
        # TODO: We don't need to worry about data location since generated code should automatically know the data location.
        # TODO: When do we change where the start of CC1 and CC2 data is located?
        for item in sorted( params.get(T_IN_OUT, {}), key=lambda x: sizes.get(params[T_IN_OUT][x]['type'], 0) if sizes else 1, reverse=True ):
            location = params[T_IN_OUT][item]['location']
            write_section_pointers( T_IN_OUT, item, 'start-in', 'end-in', data_copy, file )
            data_copy.append(f"\t\tpinnedPtrs_[n].{location}_data = static_cast<Real*>( static_cast<void*>(char_ptr) );\n\n")
        file.write(f"\t// end copy in out\n\n")
        ### 

        ### DETERMINE COPY OUT POINTERS
        file.write(f"\t// copy out section\n")
        out_location = []
        if T_OUT in params:
            file.writelines([
                f"{indent}char* copyOutStart_p = copyInOutStart_p_ + {T_IN_OUT.replace('-', '')}DataBytesPadded;\n",
                f"{indent}char* copyOutStart_d = copyInOutStart_d_ + {T_IN_OUT.replace('-', '')}DataBytesPadded;\n"
                f"{indent}ptr_p = copyOutStart_p;\n"
                f"{indent}ptr_d = copyOutStart_d;\n\n"
            ])
        for item in sorted( params.get(T_OUT, {}), key=lambda x: sizes.get(params[T_OUT][x]['type'], 0) if sizes else 1, reverse=True ):
            location = params[T_OUT][item]['location']
            write_section_pointers(T_OUT, item, 'start', 'end', out_location, file)
            out_location.append(f"\t\tpinnedPtrs_[n].{location}_data = static_cast<Real*>( static_cast<void*>(char_ptr) );\n\n")
        file.write(f"\t// end copy out\n\n")
        ###
        file.write(f"\t/// END\n\n")

        file.writelines([
            f"\tif (pinnedPtrs_) throw std::logic_error(\"{packet_name}::pack Pinned pointers already exist\");\n",
            f"\tpinnedPtrs_ = new BlockPointersPinned[{N_TILES}];\n",
            "\tPacketContents* tilePtrs_p = contents_p_;\n",
            "\tchar* char_ptr;\n",
            "\t/// MEM COPY SECTION\n",
            ] + general_copy_in
            + ["\n"]
        )
            
        # tile specific metadata.
        file.write(f"\tfor ({SIZE_T} n=0; n < {N_TILES}; ++n{', ++tilePtrs_p' if args.language == mdata.Language.cpp else ''}) {{\n")
        file.writelines([
            f"\t\tTile* tileDesc_h = tiles_[n].get();\n",
            f"\t\tif (tileDesc_h == nullptr) throw std::runtime_error(\"[{packet_name}::{func_name}] Bad tileDesc.\");\n",
        ])

        metadata_list = params.get(T_MDATA, [])
        # NOTE: This is required for cpp packets since the data arrays require all 4 of lo, hi, loGC and hiGC
        # This is not required for fortran packets since the data arrays are not converted to FArray4Ds.
        if args.language == mdata.Language.cpp:
            dependencies = set(params[T_MDATA]).symmetric_difference( {"lo", "hi", "loGC", "hiGC"} ).intersection({"lo", "hi", "loGC", "hiGC"})
            metadata_list = set(params.get(T_MDATA, [])).union(dependencies)
        for item in metadata_list:
                item_type = mdata.tile_known_types[item]
                # if item_type not in 
                file.write(f"\t\tconst {item_type} {item} = {TILE_DESC}->{item}();\n")

        file.writelines([
            f"\t\tReal* data_h = {TILE_DESC}->dataPtr();\n"
            f"\t\tif (data_h == nullptr) throw std::logic_error(\"[{packet_name}::{func_name}] Invalid ptr to data in host memory.\");\n\n"
        ])

        # TODO: Could we possibly merge T_MDATA and the device_array_pointers sections?
        for item in sorted(params.get(T_MDATA, []), key=lambda x: sizes.get(mdata.tile_known_types[x], 0) if sizes else 1, reverse=True):
            src = f"&{item}"
            file.write(f"\t\tchar_ptr = static_cast<char*>({item}{START_P}) + n * {item}{BLOCK_SIZE};\n" )
            if args.language != mdata.Language.cpp:
                if "vect" in mdata.tile_known_types[item].lower(): #array type
                    offset = " + 1" if 'int' in mdata.tile_known_types[item].lower() else ""
                    file.write(f'\t\t{mdata.cpp_equiv[mdata.tile_known_types[item]]} {item}_h[MILHOJA_MDIM] = {{{item}.I(){offset}, {item}.J(){offset}, {item}.K(){offset}}};\n')
                    src = f"{item}_h"
                else: # primitive
                    ty = mdata.tile_known_types[item].replace('unsigned ', '')
                    file.write(f"\t\t{ty} {item}_h = static_cast<{ty}>({item});\n")
                    src = f"&{item}_h"
            else:
                file.write(f"\t\ttilePtrs_p->{item}_d = static_cast<{mdata.tile_known_types[item]}*>(static_cast<void*>(char_ptr));\n")
            file.write(f"\t\tstd::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>({src}), {item}{BLOCK_SIZE});\n\n")
        file.writelines(data_copy + out_location)

        # only store farray pointers if user is using the fortran binding classes
        if args.language == mdata.Language.cpp:
            for item in sorted(device_array_pointers, key=lambda x: sizes[device_array_pointers[x]['type']] if sizes else 1, reverse=True ):
                d = 4 # assume d = 4 for now.
                section = device_array_pointers[item]['section']
                data_type = device_array_pointers[item]['type']
                location = device_array_pointers[item]['location']
                nunkvars = device_array_pointers[item][NUNK]
                indexer = device_array_pointers[item][INDEXER]
                # This is temporary until we get data locations sorted out
                c_args = mdata.finterface_constructor_args[indexer] if indexer else mdata.finterface_constructor_args[location.lower()]

                file.writelines([
                    f"\t\tchar_ptr = {location}_farray_start_d_ + n * sizeof(FArray4D);\n",
                    f"\t\ttilePtrs_p->{location}_d = static_cast<FArray{d}D*>( static_cast<void*>(char_ptr) );\n",
                    f"\t\tFArray{d}D {item}_d{{ static_cast<{data_type}*>( static_cast<void*>( static_cast<char*>({item}{START_D}) + n * {item}{BLOCK_SIZE} ) ), {c_args}, {nunkvars}}};\n"
                    f"\t\tchar_ptr = {location}_farray_start_p_ + n * sizeof(FArray4D);\n"
                    # f"\t\tchar_ptr = static_cast<char*>({item}{START_P}) + n * {item}{BLOCK_SIZE};\n"
                    f"\t\tstd::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&{item}_d), sizeof(FArray{d}D));\n\n",
                ])

        # request stream at end
        file.writelines([
            "\t}\n",
            "\t/// END\n\n"
            "\tstream_ = RuntimeBackend::instance().requestStream(true);\n",
            f"\tif (!stream_.isValid()) throw std::runtime_error(\"[{packet_name}::pack] Unable to acquire stream\");\n",
        ])

        # TODO: CHange to use the array implementation.
        file.writelines([(
            f"\tstream{i}_ = RuntimeBackend::instance().requestStream(true);\n" \
            f"\tif (!stream{i}_.isValid()) throw std::runtime_error(\"[{packet_name}::{func_name}] Unable to acquire stream {i}.\");\n")
                for i in range(2, params.get(EXTRA_STREAMS, 0)+2) ] + 
            [f"}}\n\n"]
        )
    
    # Generate clone method
    def generate_clone(file: TextIO, params: dict):
        """
        Generates the object cloning method for the data packet.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary
        Returns:
            None
        """
        packet_name = params["name"]
        file.writelines([
            f"std::unique_ptr<milhoja::DataPacket> {packet_name}::clone(void) const {{\n",
            f"\treturn std::unique_ptr<milhoja::DataPacket>{{ new {packet_name}{{{', '.join( f'{item[0]}{HOST}' for item in constructor_args)}}} }};\n"
            f"}}\n\n"
        ])

    # TODO: Right now the code generates stream{n}_ for n extra streams.
    #       Ideally, we would store all of those in an array and use indexing to check the specific id.
    #       This is what hydro variant2 actually does, but since variant 1 and variant 3 use this system,
    #       I will manually generate and check all streams until we can write our own task functions,
    #       since releasing extra queues is called outside of the datapacket object.
    def generate_release_queues(file: TextIO, params: dict):
        """
        Generates the release queue methods for the data packet.
        No methods are generated if the number of specified extra streams (n-extra-streams) is 0.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary

        Returns:
            None
        """
        packet_name = params['name']
        extra_streams = params.get(EXTRA_STREAMS, 0)
        indent = '\t'
        # extra async queues
        # only generate extra functions if we have more than 1 stream
        if extra_streams > 0:
            func_name = "extraAsynchronousQueue"
            file.writelines([
                f"int {packet_name}::{func_name}(const unsigned int id) {{\n",
                f"{indent}if ((id < 2) || (id > EXTRA_STREAMS + 1))\n"
                f"{indent*2}throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id.\");\n",
                f"{indent}switch(id) {{\n"
            ])
            file.writelines([ 
                f"{indent * 2}case {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error(\"[{packet_name}::{func_name}] Stream {i} invalid.\"); }} return stream{i}_.accAsyncQueue;\n" for i in range(2, extra_streams+2)
            ])
            file.write(f"{indent}}}\n{indent}return 0;\n}}\n\n")

            # Release extra queue
            func_name = "releaseExtraQueue"
            file.writelines([
                f"void {packet_name}::{func_name}(const unsigned int id) {{\n",
                f"{indent}if ((id < 2) || (id > EXTRA_STREAMS + 1))\n"
                f"{indent*2}throw std::invalid_argument(\"[{packet_name}::{func_name}] Invalid id.\");\n",
                f"{indent}switch(id) {{\n"
            ])
            file.writelines([
                f"{indent * 2}case {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error(\"[{packet_name}::{func_name}] Extra queue invalid. ({i})\"); }} milhoja::RuntimeBackend::instance().releaseStream(stream{i}_); break;\n" for i in range(2, extra_streams+2)
            ])
            file.write(f"{indent}}}\n}}\n\n")

    def generate_tile_size_host(file: TextIO, params: dict):
        """
        Generate the tile size host function for calculating extents and sizes.

        Parameters:
            file - The file to write to\n
            params - The JSON dictionary

        Returns:
            None
        """
        packet_name = params["name"]
        file.writelines([
            f"void {packet_name}::tileSize_host(int* nxbGC, int* nybGC, int* nzbGC, int* nCcVars, int* nFluxVars) const {{\n",
            f"\tusing namespace milhoja;\n",
            f"\t*nxbGC = static_cast<int>(nxb_ + 2 * nGuard_ * MILHOJA_K1D);\n",
            f"\t*nybGC = static_cast<int>(nyb_ + 2 * nGuard_ * MILHOJA_K2D);\n",
            f"\t*nzbGC = static_cast<int>(nzb_ + 2 * nGuard_ * MILHOJA_K3D);\n",
            f"\t*nCcVars = nCcVars_ - 8;\n",
            f"\t*nFluxVars = nFluxVars_;\n",
            f"}}\n\n"
        ])

    if not parameters:
        raise ValueError("Parameters is empty or null.")

    name = parameters["file_name"]
    with open(f"{name}.cpp", "w") as code:
        # We might need to include specific headers based on the contents of the json packet
        code.writelines([
            GENERATED_CODE_MESSAGE, f"#include \"{os.path.basename(name)}.h\"\n",
            f"#include <cassert>\n", f"#include <cstring>\n",
            f"#include <stdexcept>\n", f"#include <Milhoja_Grid.h>\n",
            f"#include <Milhoja_RuntimeBackend.h>\n"
        ])

        generate_constructor(code, parameters)
        generate_destructor(code, parameters)
        generate_release_queues(code, parameters)
        generate_clone(code, parameters)
        generate_tile_size_host(code, parameters)
        generate_pack(code, parameters, args)
        generate_unpack(code, parameters)

def generate_cpp_header_file(parameters: dict, args):
    """
    Generates the header file for the data packet.

    Parameters:
            file - The file to write to\n
            params - The JSON dictionary\n
            args - The namespace containing the command line arguments

    Returns:
        None
    """
    if not parameters:
        raise ValueError("Parameters is null.")

    with open(parameters["file_name"] + ".h", "w") as header:
        name = parameters["name"]
        types = set("FArray4D") if args.language == mdata.Language.cpp else set()
        header.writelines([
            GENERATED_CODE_MESSAGE,
            f"#ifndef {name.upper()}_\n",
            f"#define {name.upper()}_\n",
            "#include <Milhoja.h>\n",
            "#include <Milhoja_DataPacket.h>\n",
        ])

        # manually generate nTiles getter here
        private_variables = [ 
            f"\tint {N_TILES};\n\tvoid* nTiles{START_P} = 0;\n", 
            f"\tvoid* nTiles{START_D} = 0;\n" 
        ]
        getters = [
            f"\tint* nTiles(void) const {{ return static_cast<int*>(nTiles{START_D}); }}\n",
            f"\tint nTiles_host(void) const {{ return {N_TILES}; }}\n"
        ]

        # Everything in the packet consists of pointers to byte regions
        # so we make every variable a pointer
        # TODO: What if we want to put array types in any section?
        # TODO: Assume FArray4D by default for now.
        def parse_general():
            """Parses the general section of the JSON."""
            general = parameters.get(GENERAL, [])
            for item in general:
                private_variables.extend([ 
                    f"\tvoid* {item}{START_P} = 0;\n",
                    f"\tvoid* {item}{START_D} = 0;\n"
                ])
                item_type = general[item] if not is_enumerable_type(general[item]) else general[item]['type']
                types.add(item_type)
                if item_type in mdata.imap: item_type = f"milhoja::{item_type}"
                constructor_args.append([item, f"const {item_type}"])
                # be careful here we can't assume all items in general are based in milhoja
                ext = "milhoja::" if item_type in mdata.imap else ""
                getters.append(f"\t{ext}{item_type}* {item}(void) const {{ return static_cast<{ext}{item_type}*>({item}{START_D}); }}\n")

        # Generate private variables for each section. Here we are creating a size helper
        # variable for each item in each section based on the name of the item
        def parse_mdata():
            """Parses the metadata section of the JSON."""
            for item in parameters.get(T_MDATA, []):
                item_type = mdata.tile_known_types[item]
                private_variables.extend([
                    f"\tvoid* {item}{START_P} = 0;\n",
                    f"\tvoid* {item}{START_D} = 0;\n"
                ])

                if not mdata.tile_known_types[item]:
                    warnings.warn("Bad data found in tile-metadata. Continuing...")
                    continue
                types.add( item_type )
                if args.language != mdata.Language.cpp and item_type in mdata.cpp_equiv:
                    item_type = mdata.cpp_equiv[item_type]
                ext = "milhoja::" if item_type in mdata.imap else ""
                item_type = item_type.replace("unsigned ", "")
                getters.append(f"\t{ext}{item_type}* {item}(void) const {{ return static_cast<{ext}{item_type}*>({item}{START_D}); }}\n")

        def parse_array_sections():
            """Parses the array sections of the JSON. (scratch, t-in, t-in-out, t-out)"""
            for sect in [T_IN, T_IN_OUT, T_OUT, T_SCRATCH]:
                for item in parameters.get(sect, {}):
                    start = parameters[sect][item][f'{START}-in' if sect == T_IN_OUT else START]
                    end = parameters[sect][item][f'{END}-in' if sect == T_IN_OUT else END]
                    item_type = parameters[sect][item]['type']
                    ext = parameters[sect][item]['extents']
                    extents, nunkvar, indexer, num_elems = mdata.parse_extents(ext, start, end, item_type, args.language)
                    parameters[sect][item][EXPANDED] = extents
                    parameters[sect][item][NUNK] = nunkvar
                    parameters[sect][item][INDEXER] = indexer
                    parameters[sect][item][NUM_ELEMS] = num_elems
                    if 'location' in parameters[sect][item]: farray_items.append(parameters[sect][item]['location'])
                    types.add(item_type)
                    device_array_pointers[item] = {"section": sect, **parameters[sect][item]}
                    private_variables.append(f"\tvoid* {item}{START_D} = 0;\n")
                    if sect != T_SCRATCH:
                        private_variables.append(f"\tvoid* {item}{START_P} = 0;\n")
                    
                    ext = "milhoja::" if item_type in mdata.imap else ""
                    getters.append(f"\t{ext}{item_type}* {item}(void) const {{ return static_cast<{ext}{item_type}*>({item}{START_D}); }}\n")

        parse_general()
        parse_mdata()
        parse_array_sections()

        private_variables.extend([
            "\tunsigned int nxb_ = 1;\n", "\tunsigned int nyb_ = 1;\n", "\tunsigned int nzb_ = 1;\n",
            "\tconst unsigned int nGuard_;\n", "\tconst unsigned int nCcVars_;\n", "\tconst unsigned int nFluxVars_;\n"
        ])

        initialize.update({
            'nGuard_': 'milhoja::Grid::instance().getNGuardcells()',
            'nCcVars_': 'milhoja::Grid::instance().getNCcVariables()',
            'nFluxVars_': 'milhoja::Grid::instance().getNFluxVariables()'
        })

        # Constructors & = operations
        header.writelines([
            ''.join( f"#include {mdata.imap[item]}\t\n" for item in types if item in mdata.imap),
            f"class {name} : public milhoja::DataPacket {{ \n",    # class definition
            "public:\n",   # public information
            f"\tstd::unique_ptr<milhoja::DataPacket> clone(void) const override;\n",
            f"\t{name}({', '.join(f'{item[1]} {NEW}{item[0]}' for item in constructor_args)});\n",
            f"\t~{name}(void);\n",
            f"\t{name}({name}&)                  = delete;\n",
            f"\t{name}(const {name}&)            = delete;\n",
            f"\t{name}({name}&& packet)          = delete;\n",
            f"\t{name}& operator=({name}&)       = delete;\n",
            f"\t{name}& operator=(const {name}&) = delete;\n",
            f"\t{name}& operator=({name}&& rhs)  = delete;\n",
            "\tvoid tileSize_host(int* nxbGC, int* nybGC, int* nzbGC, int* nCcVars, int* nFluxVars) const;\n",
            "\tvoid pack(void) override;\n",
            "\tvoid unpack(void) override;\n",
            "".join(getters)
        ])

        extra_streams = parameters.get(EXTRA_STREAMS, 0)
        if extra_streams > 0: # don't need to release extra queues if we only have 1 stream.
            header.writelines([
                "\tint extraAsynchronousQueue(const unsigned int id) override;\n",
                "\tvoid releaseExtraQueue(const unsigned int id) override;\n",
                "private:\n",
                f"\tstatic const unsigned int EXTRA_STREAMS = {extra_streams};\n"
            ] + [f'\tmilhoja::Stream stream{i}_;\n' for i in range(2, extra_streams+2)])
        else:
            header.write("private:\n")

        header.writelines([
            f"\tstatic constexpr std::size_t ALIGN_SIZE={parameters.get('byte-align', 16)};\n",
            "\tstatic constexpr std::size_t pad(const std::size_t size) {{ return ((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE; }}\n",
            ''.join( f'\t{item[1]} {item[0]}{HOST};\n' for item in constructor_args),
            ''.join(private_variables),
            "};\n",
            "#endif\n"
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate packet code files for use in Flash-X simulations.")
    parser.add_argument("JSON", help="The JSON file to generate from.")
    parser.add_argument('--language', '-l', type=mdata.Language, choices=list(mdata.Language), help="Generate a packet to work with this language.")
    parser.add_argument("--sizes", "-s", help="Path to data type size information.")
    args = parser.parse_args()

    if args.language is None:
        print("Language not specified. Packet will not be generated.")
        exit(0)

    if ".json" not in os.path.basename(args.JSON):
        print("Provided file is not a JSON file.")
        exit(-1)
    elif os.path.getsize(args.JSON) < 5:
        print("JSON file is empty, packet will not be generated.")
        exit(0)
    with open(args.JSON, "r") as file:
        data = json.load(file)
        data["file_name"] = file.name.replace(".json", "")
        data["name"] = os.path.basename(file.name).replace(".json", "")
        clean = mdata.check_task_argument_list(data)
        if not clean:
            print("There are duplicates in task argument order or the the order list does not match the arguments.")
            exit(-1)

        # import timeit
        # t = timeit.Timer(lambda: generate_cpp_header_file(data, args))
        # t2 = timeit.Timer(lambda: generate_cpp_code_file(data, args))
        generate_cpp_header_file(data, args)
        generate_cpp_code_file(data, args)
        # print(t.timeit(10000), t2.timeit(10000))

