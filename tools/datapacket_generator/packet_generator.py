# Author: Wesley Kwiecinski

# Packet generator for Milhoja. Script takes in a 
# JSON file and generates cpp code for a data packet
# based on the contents of the json file.
# 
# TODO: We should also be adding support for generating fortran file packets in the future.
# 
# TODO: Work on documentation for packet json format. Refer to DataPacketGeneratorDoc for documentation

import sys
import json

GENERATED_CODE_MESSAGE = "// This code was generated with packet_generator.py.\n"

# All possible sections.
GENERAL = "general"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_IN_OUT = "tile-in-out"
T_OUT = "tile-out"

# 

# type constants
SIZE_T = "std::size_t"
BLOCK_SIZE = "_BLOCK_SIZE_HELPER"
N_TILES = "nTiles"
SCRATCH = "_scratch_d"
TILE_DESC = "tileDesc_h"
DATA_P = "_data_p"
DATA_D = "_data_d"
IN_OUT = "in_out"
OUT = "out"
# 

# these might not be necessary
vars_and_types = {}
level = 0

# TODO: Maybe we can create dictionaries derived from sections in the json? Something to think about
known_sections = set()

# TODO: It might be beneficial to write helper methods or a wrapper class for files
# to help with consistently writing to the file, so we
# don't have to put {indent} or \n in every line we write.

def get_indentation(level):
    return "\t" * level

def generate_cpp_file(parameters):

    def generate_constructor(file, params):
            # function definition
            file.write("%s::%s(milhoja::Real dt = nullptr) : milhoja::DataPacket(){}, \n" % (params["name"], params["name"]))
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
        indent = '\t'
        if params["ndim"] == 3:
            file.write(f"{indent}if (stream2_.isValid() || stream3_.isValid()) throw std::logic_error(\"[DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3] One or more extra streams not released\");")

        file.write(f"\n}}\n\n")
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
            f"{indent}if (!stream.isValid()) throw std::logic_error(\"[{packet_name}::{func_name}] Stream not acquired.\");\n",
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
            nunkvars = dict_to_use[item]['extents'][-1]
            type = dict_to_use[item]['type']
            # TODO: The way the constructor and header files we need to do some division to 
            # get the origin num vars per CC per variable. This is a way to do it without creating
            # another variable. Low priority
            file.writelines([
                f"{indent}{SIZE_T} offset = {item}{BLOCK_SIZE} * (1 / {nunkvars}) * (1 / sizeof({type})) * static_cast<{SIZE_T}>(startVariable_);\n",
                f"{indent}{type}* start_h = data_h + offset;\n"
                f"{indent}const {type}* start_p = data_p + offset;\n"
                f"{indent}{SIZE_T} nBytes = (endVariable_ - startVariable_ + 1) * ({item}{BLOCK_SIZE} * (1 / {nunkvars}));\n"
                f"{indent}std::memcpy((void*)start_h, (void*)start_p, nBytes);\n"                
            ])

        indent = '\t'

        file.write(f"{indent}}}\n")

        file.write(f"}}\n")
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
            f"{indent}if (errMsg != \"\") {{\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent} else if (tiles_.size() == 0) {{\n",
            f"{indent*2}throw std::logic_error(\"[{packet_name}::{func_name}] \" + errMsg);\n"
            f"{indent}}}\n"
            f"{indent}Grid& grid = Grid::instance();\n"
        ])

        # # Scratch section generation.
        file.write(f"{indent}// Scratch section\n")

        array_section_map = {}

        # TODO: Note: Tile-out does NOT double as a scratch section, the pdf was just incorrect. Let's fix the pdf.
        nScratchArrs = 0
        file.write(f"{indent}{SIZE_T} nScratchPerTileBytes = 0")
        for var in params.get(T_SCRATCH, []):
            nScratchArrs += 1
            file.write(f" + {var}{BLOCK_SIZE}")
            array_section_map[var] = {"section": SCRATCH, **params[T_SCRATCH][var]}
        file.write(f";\n{indent}unsigned int nScratchArrays = {nScratchArrs};\n")

        bytesToGpu = set()
        returnToHost = set()
        bytesPerPacket = set()
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

        # nBlockMetadata, we do 5 * sizeof(FArray4D) since there are 5 associated scratch arrays per tile. Maybe this is something we can change.
        file.write(f"{indent}{SIZE_T} nBlockMetadataPerTileBytes = 5 * sizeof(FArray4D)")
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
                array_section_map[item] = {'section': DATA_D, **cin[item]}
            bytesToGpu.add("(nTiles * nCopyInDataPerTileBytes)")
            bytesPerPacket.add("(nTiles * nCopyInDataPerTileBytes)")
        file.write(f";\n")

        # copy in out data
        cinout = params.get(T_IN_OUT, {})
        file.write(f"{indent}{SIZE_T} nCopyInOutDataPerTileBytes = 0")
        if cinout:
            for item in cinout:
                file.write(f" + {item}{BLOCK_SIZE}")
                array_section_map[item] = {'section': T_IN_OUT, **cinout[item]}
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
                array_section_map[item] = {'section': T_OUT, **cout[item]}
            bytesToGpu.add("(nTiles * nCopyOutDataPerTileBytes)")
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

        # request streams
        file.writelines([
            f"{indent}stream_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}if (!stream_.isValid()) {{\n",
            f"{indent * 2}throw std::runtime_error(\"[{packet_name}::pack] Unable to acquire stream\");\n{indent}}}\n"
            f"# if MILHOJA_NDIM == 3\n", # we can get rid of the compiler directives eventually.
            f"{indent}stream2_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}stream3_ = RuntimeBackend::instance().requestStream(true);\n",
            f"{indent}if (!stream2_.isValid() || !stream3_.isValid()) {{\n",
            f"{indent * 2}throw std::runtime_error(\" [{packet_name}::pack] Unable to acquire extra streams\");\n{indent}}}\n#endif\n"
        ])

        # # acquire gpu mem
        # Note that copyin, copyinout, and copyout all have a default of 0. So to keep code generation easy
        # we set them to the default of 0 even if they don't exist in the json.
        file.write(f"{indent}RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - {N_TILES} * nScratchPerTileBytes, &packet_p_, nBytesPerPacket, &packet_d_);\n")
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
                f"{indent}char* copyOutStart_p = copyInOutStart_p;\n",
                f"{indent}char* copyOutStart_d = copyInOutStart_d;\n"
            ])

        file.writelines([
            f"{indent}if (pinnedPtrs_) throw std::logic_error(\"{packet_name}::pack Pinned pointers already exist\");\n",
            f"{indent}pinnedPtrs_ = new BlockPointersPinned[{N_TILES}];\n"
        ]) 

        # scratch section? Again?

        # copy-in section
        file.writelines([
            f"{indent}static_assert(sizeof(char) == 1, \"Invalid char size\");\n", # we might not need this anymore
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

        tile_ptrs = {}
        if T_IN in params:
            file.write(f"{indent}char* { '_'.join(params[T_IN].keys()) }{DATA_P} = copyInStart_p_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            file.write(f"{indent}char* { '_'.join(params[T_IN].keys()) }{DATA_D} = copyInStart_d_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            tile_ptrs[next(iter(params[T_IN]))] = T_IN_OUT

        # There's probably going to only be 1 item per tile-in, tile-in-out, and tile-out.
        # TODO: The variants for each packet don't have CC1 set co copyInOut.
        # TODO: Why is pinnedPtrs_[n].CC1_data set to nullptr and other times it isn't? Might want to email jared about this
        # TODO: When do we change where the start of CC1 and CC2 data is located?
        # for item in params.get(T_IN_OUT, {}):
        if T_IN_OUT in params:
            file.write(f"{indent}char* { '_'.join(params[T_IN_OUT].keys()) }{DATA_P} = copyInStart_p_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            file.write(f"{indent}char* { '_'.join(params[T_IN_OUT].keys()) }{DATA_D} = copyInStart_d_ + nCopyInBytes + ({N_TILES} * nBlockMetadataPerTileBytes);\n")
            tile_ptrs[next(iter(params[T_IN_OUT]))] = T_IN_OUT
         
        # for item in params.get(T_OUT, {}):
        if T_OUT in params:
            file.write(f"{indent}char* { '_'.join(params[T_OUT].keys()) }{DATA_P} = copyOutStart_p;\n")
            file.write(f"{indent}char* { '_'.join(params[T_OUT].keys()) }{DATA_D} = copyOutStart_d;\n")
            tile_ptrs[next(iter(params[T_OUT]))] = T_OUT

        # Create all scratch ptrs.
        scr = sorted(list(params.get(T_SCRATCH, {}).keys()))
        for i in range(0, len(scr)):
            if i == 0:  # we can probably use an iterator here instead
                file.write(f"{indent}char* {scr[i]}{SCRATCH} = scratchStart_d;\n")
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

        if T_IN in params:
            file.write(f"{indent}std::memcpy((void*){'_'.join(params[T_IN])}{DATA_P}, (void*)data_h, 0")
            for item in params[T_IN]:
                file.write(f" + {item}{BLOCK_SIZE}")
        elif T_IN_OUT in params:
            file.write(f"{indent}std::memcpy((void*){'_'.join(params[T_IN_OUT])}{DATA_P}, (void*)data_h, 0")
            for item in params[T_IN_OUT]:
                file.write(f" + {item}{BLOCK_SIZE}")
        file.write(f");\n")

        # be careful here, is pinnedptrs tied to tile-in-out or tile-out? What about tile-in?
        if T_IN_OUT in params:
            nxt = next(iter(params[T_IN_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = static_cast<{params[T_IN_OUT][nxt]['type']}*>((void*){ '_'.join(params[T_IN_OUT].keys()) }{DATA_P});\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC1_data = nullptr;\n")

        if T_OUT in params:
            nxt = next(iter(params[T_OUT]))
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = static_cast<{params[T_OUT][nxt]['type']}*>((void*){ '_'.join(params[T_OUT].keys()) }{DATA_P});\n\n")
        else:
            file.write(f"{indent}pinnedPtrs_[n].CC2_data = nullptr;\n\n")

        possible_tile_ptrs = ['deltas', 'lo', 'hi', 'CC1', 'CC2', 'FCX', 'FCY', 'FCZ']
        # Add metadata to ptr
        for item in params.get(T_MDATA, []):
            possible_tile_ptrs.remove(item)
            file.writelines([
                f"{indent}tilePtrs_p->{item}_d = static_cast<{params[T_MDATA][item]}*>((void*)ptr_d);\n",
                f"{indent}std::memcpy((void*)ptr_p, (void*)&{item}, {item}{BLOCK_SIZE});\n",
                f"{indent}ptr_p += {item}{BLOCK_SIZE};\n"
                f"{indent}ptr_d += {item}{BLOCK_SIZE};\n\n"
            ])

        # Tile ptrs
        array_section_map = dict(sorted(array_section_map.items()))
        # print(array_section_map)
        print_type = True
        for item in array_section_map:
            section = array_section_map[item]['section']
            file.write(f"{indent}tilePtrs->{item}_d = static_cast<FArray4D*>((void*)ptr_d);\n"), # set tile ptrs cc ptr.)
            if section == SCRATCH:
                if 'CC' in item:    # Cell centered data has specific format requirements
                    type = array_section_map[item]['type']
                    unk = array_section_map[item]['extents'][-1]
                    file.writelines([
                        f"{indent}FArray4D {item}_d{{ static_cast<{type}*>((void*){item}{SCRATCH}), loGC, hiGC, {unk}}};\n",
                        f"{indent}std::memcpy((void)*ptr_p, (void*)&{item}_d, sizeof(FArray4D));\n",
                        f"{indent}ptr_p += sizeof(FArray4D);\n",
                        f"{indent}ptr_d += sizeof(FArray4D);\n",
                        f"{indent}{item}{SCRATCH} += nScratchPerTileBytes;\n\n"
                    ])
                else:   # Face array.
                    # I don't like doing this
                    fHi = "{LIST_NDIM(hi.I()+1, hi.J(), hi.K())}"
                    if item == 'FCY': fHi = "{LIST_NDIM(hi.I(), hi.J()+1, hi.K())}"
                    elif item == 'FCZ': fHi = "{LIST_NDIM(hi.I(), hi.J(), hi.K())+1}" 

                    type = array_section_map[item]['type']
                    unk = array_section_map[item]['extents'][-1]
                    file.writelines([
                        f"{indent}IntVect {item}_fHi = IntVect{ fHi };\n",
                        f"{indent}Farray4D {item}_d{{ static_cast<{type}*>((void*){item}{SCRATCH}), lo, {item}_fHi, {unk}}};\n"
                        f"{indent}std::memcpy((void)*ptr_p, (void*)&{item}_d, sizeof(FArray4D));\n",
                        f"{indent}ptr_p += sizeof(FArray4D);\n",
                        f"{indent}ptr_d += sizeof(FArray4D);\n",
                        f"{indent}{item}{SCRATCH} += nScratchPerTileBytes;\n\n"
                    ])
            else:
                print(section)
                type = array_section_map[item]['type']
                unk = array_section_map[item]['extents'][-1]
                file.writelines([   # careful with naming here.
                    f"{indent}FArray4D {item}_d{{ static_cast<{type}*>((void*){item}{DATA_D}), loGC, hiGC, {unk}}};\n",
                    f"{indent}std::memcpy((void)*ptr_p, (void*)&{item}_d, sizeof(FArray4D));\n",
                    f"{indent}ptr_p += sizeof(FArray4D);\n",
                    f"{indent}ptr_d += sizeof(FArray4D);\n",
                    f"{indent}{item}{DATA_P} += {item}{BLOCK_SIZE};\n"
                    f"{indent}{item}{DATA_D} += {item}{BLOCK_SIZE};\n\n"
                ])
            print_type = False
            possible_tile_ptrs.remove(item)

        # if there are unremoved items we set them to nullptr.
        for item in possible_tile_ptrs:
            file.write(f"{indent}tilePtrs_p->{item}_d = nullptr")

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
    ndim = parameters["ndim"]
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
        
        if ndim == 3:
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
        ndim = parameters["ndim"]
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
        header.write(indent + f"{name}(milhoja::Real dt = nullptr);\n")
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

        if ndim == 3:
            header.writelines([
                f"{indent}int extraAsynchronousQueue(const unsigned int id) override;\n",
                f"{indent}void releaseExtraQueue(const unsigned int id) override;\n"
            ])
            
        # queue methods
        # header.writelines([
        #     f"#if MILHOJA_NDIM == 3 && defined(MILHOJA_OPENACC_OFFLOADING)\n",
        #     f"{indent}int extraAsynchronousQueue(const unsigned int id) override;\n",
        #     f"{indent}void releaseExtraQueue(const unsigned int id) override;\n"
        #     f"#endif\n"
        # ])
    
        # private information
        header.write("private:\n")
        if ndim == 3: # check ndim for adding extra streams
            header.writelines([
                f"{indent}milhoja::Stream stream2_;\n",
                f"{indent}milhoja::Stream stream3_;\n"
            ])

        # header.writelines([
        #     f"#if MILHOJA_NDIM==3\n",
        #     f"{indent}milhoja::Stream stream2_;\n",
        #     f"{indent}milhoja::Stream stream3_;\n",
        #     "#endif\n"
        # ])

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

        if T_IN_OUT in parameters:
            known_sections.add(T_IN_OUT)
            for item in parameters[T_IN_OUT]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        if T_OUT in parameters:
            known_sections.add(T_OUT)
            for item in parameters[T_OUT]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        if T_SCRATCH in parameters:
            known_sections.add(T_SCRATCH)
            for item in parameters[T_SCRATCH]:
                header.write(f"{indent}{SIZE_T} {item}{BLOCK_SIZE};\n")
                vars_and_types[f"{item}{BLOCK_SIZE}"] = SIZE_T

        level -= 1
        indent = get_indentation(level)
        header.write("};\n")

        # end start define
        header.write("#endif\n")

    return

# Takes in a file path to load a json file and generates the cpp and header files
def generate_packet_with_filepath(fp):
    with open(fp, "r") as file:
        data = json.load(file)
        generate_header_file(data)
        generate_cpp_file(data)

# gneerate packet data using existing dict
def generate_packet_with_dict(json_dict):
    generate_header_file(json_dict)
    generate_cpp_file(json_dict)

if __name__ == "__main__":
    #Check if some file path was passed in
    if len(sys.argv) < 2:
        print("Usage: python packet_generator.py [data_file]")
        
    generate_packet_with_filepath(sys.argv[1])