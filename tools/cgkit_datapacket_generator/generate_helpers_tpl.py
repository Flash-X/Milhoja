"""
This is the main script for generating a new DataPacket. The overall structure of this 
code is to move through every possible DataPacket JSON section and fill out the various CGKkit dictionaries
for generating every template.

TODO: How to sort bound class members by size?
"""

from dataclasses import dataclass

_CON_ARGS = 'constructor_args'
_SET_MEMBERS = 'set_members'
_SIZE_DET = 'size_determination'
_HOST_MEMBERS = 'host_members'
_PUB_MEMBERS = 'public_members'
_PINNED_SIZES = 'pinned_sizes'
_IN_PTRS = 'in_pointers'
_OUT_PTRS = 'out_pointers'
_T_DESCRIPTOR = 'tile_descriptor'

_STREAM_FUNCS_H = 'stream_functions_h'
_EXTRA_STREAMS = 'extra_streams'
_DESTRUCTOR = 'destructor'
_STREAM_FUNCS_CXX = 'stream_functions_cxx'

from collections import defaultdict
import packet_generation_utility as util
import warnings
import cpp_helpers
import json_sections as jsc
import DataPacketMemberVars as dpinfo

def add_size_parameter(name: str, section_dict: dict, connectors: dict):
    """
    Adds a size connector to the params dict that is passed in. The size connector dictionary stores every size 
    of every item in the data packet for use with a CGKit template.

    :param str name: The name of the size connector. Usually the name of a section.
    :param dict section_dict: The section used to generate sizes. Contains all of the items in the section.
    :param dict connectors: The dictionary containing all connectors generated for the packet.
    :rtype: None
    """
    connectors[f'size_{name}'] = ' + '.join( f'SIZE_{item.upper()}' for item in section_dict) if section_dict else '0'

def section_creation(name: str, section: dict, connectors: dict, size_connectors):
    """
    Creates a section and sets the default value to an empty list. It's assumed that the function that calls this method 
    will populate the dictionary using the same name.
    
    :param str name: The name of the section to create.
    :param dict section: The dictionary to get all data packet items from. 
    """
    add_size_parameter(name, section, size_connectors)
    connectors[f'pointers_{name}'] = []
    connectors[f'memcpy_{name}'] = []

def set_pointer_determination(connectors: dict, section: str, info: dpinfo.DataPacketMemberVars, item_is_member_variable=True):
    """
    Stores the code for determining pointer offsets in the DataPacket into the connectors dictionary to be used with CGKit.
    
    :param dict connectors: The connectors dictionary containing all connectors needed for use with CGKit.
    :param str section: The name of the section to use to determine the key for the pointer determination.
    :param DataPacketMemberVars info: Contains information for formatting the name to get variable names.
    :param bool item_is_member_variable: Flag if *item* pinned memory pointer is not a member variable, defaults to True 
    """
    dtype = info.dtype
    # If the item is not a data packet member variable, we don't need to specify a type name here.
    # Note that pointers to memory in the remote device are always member variables, so it will never need a type name.
    if not item_is_member_variable: dtype = ""
    else: dtype += "* "
    
    connectors[f'pointers_{section}'].append(
        f"""{dtype}_{info.ITEM}_p = static_cast<{info.dtype}*>( static_cast<void*>(ptr_p) );\n""" + 
        f"""{info.get_device()} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
        f"""ptr_p+={info.get_size(info.PER_TILE)};\n""" + 
        f"""ptr_d+={info.get_size(info.PER_TILE)};\n\n"""
    )

def generate_extra_streams_information(connectors: dict, extra_streams: int):
    """
    Fills the links extra_streams, stream_functions_h/cxx.

    :param dict connectors: The dictionary containing all connectors to be used with CGKit.
    :param int extra_streams: The number of extra streams specified in the data packet JSON.
    :rtype: None
    """
    if extra_streams < 1: return

    connectors[_STREAM_FUNCS_H].extend([
        f'int extraAsynchronousQueue(const unsigned int id) override;\n',
        f'void releaseExtraQueue(const unsigned int id) override;\n'
    ])
    connectors[_EXTRA_STREAMS].extend([
        f'Stream stream{i}_;\n' for i in range(2, extra_streams+2)
    ])
    connectors[_DESTRUCTOR].extend([
        f'if (stream{i}_.isValid()) throw std::logic_error("[_param:class_name::~_param:class_name] Stream {i} not released");\n'
        for i in range(2, extra_streams+2)
    ])

    connectors[_STREAM_FUNCS_CXX].extend([
            f'int _param:class_name::extraAsynchronousQueue(const unsigned int id) {{\n',
            f'\tif((id < 2) || (id > {extra_streams} + 1)) throw std::invalid_argument("[_param:class_name::extraAsynchronousQueue] Invalid id.");\n',
            '\tswitch(id) {\n'
        ] + [
            f'\t\tcase {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error("[_param:class_name::extraAsynchronousQueue] Stream {i} invalid."); }} return stream{i}_.accAsyncQueue;\n'
            for i in range(2, extra_streams+2)
        ] + [
            '\t}\n',
            '\treturn 0;\n',
            '}\n'
        ] + [
            f'void _param:class_name::releaseExtraQueue(const unsigned int id) {{\n',
            f'\tif((id < 2) || (id > {extra_streams} + 1)) throw std::invalid_argument("[_param:class_name::releaseExtraQueue] Invalid id.");\n',
            '\tswitch(id) {\n'
        ] + [
            f'\t\tcase {i}: if(!stream{i}_.isValid()) {{ throw std::logic_error("[_param:class_name::releaseExtraQueue] Stream {i} invalid."); }} milhoja::RuntimeBackend::instance().releaseStream(stream{i}_); break;\n'
            for i in range(2, extra_streams+2)
        ] + [   
            '\t}\n'
            '}\n'
    ])

    connectors[jsc.EXTRA_STREAMS].extend([
        f'stream{i}_ = RuntimeBackend::instance().requestStream(true);\n' + 
        f'if(!stream{i}_.isValid()) throw std::runtime_error("[_param:class_name::pack] Unable to acquire second stream");\n'
        for i in range(2, extra_streams+2)
    ])

def iterate_constructor(connectors: dict, size_connectors: dict, constructor: dict):
    """
    Iterates the constructor / thread-private-variables section and adds the necessary connectors.

    :param dict connectors: The dictionary containing all connectors for use with CGKit.
    :param dict size_connectors: The dictionary containing all size connectors for determining the sizes of each item in the packet.
    :param dict constructor: The dictionary containing the data for the ptv section in the DataPacket JSON.
    :rtype: None
    """
    section_creation(jsc.GENERAL, constructor, connectors, size_connectors)
    connectors[_HOST_MEMBERS] = []
    for key,item_type in constructor.items():
        info = dpinfo.DataPacketMemberVars(item=key, dtype=item_type, size_eq=f'pad( sizeof({item_type})', per_tile=False)

        if key != 'nTiles':
            connectors[_CON_ARGS].append( f'{item_type} {key}' )
            connectors[_HOST_MEMBERS].append( info.get_host() )
        connectors[_PUB_MEMBERS].extend(
            [f'{info.dtype} {info.get_host()};\n', 
             f'{info.dtype}* {info.get_device()};\n']
        )
        connectors[_SET_MEMBERS].extend(
            [f'{info.get_host()}{"{tiles_.size()}" if key == "nTiles" else f"{{{key}}}"}', 
             f'{info.get_device()}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.get_size(False)} =  {info.SIZE_EQ});\n'
        )
        set_pointer_determination(connectors, jsc.GENERAL, info)
        connectors[f'memcpy_{jsc.GENERAL}'].append(
            f'std::memcpy({info.get_pinned()}, static_cast<void*>(&{info.get_host()}), {info.get_size(False)});\n'
        )

def tmdata_memcpy_f(connectors: dict, construct: str, use_ref: str, info: dpinfo.DataPacketMemberVars, alt_name: str):
    """
    Adds the memcpy portion for the metadata section in a fortran packet.
    
    :param dict connectors: The dictionary containing all cgkit connectors.
    :param str construct: The generated host variable to copy to the pinned pointer location
    :param str use_ref: Use a reference to the host item.
    :param DataPacketMemberVars info: Contains information for formatting the name to get variable names.
    :param str alt_name: Unused
    """
    connectors[f'memcpy_{jsc.T_MDATA}'].extend([
        f'{info.dtype} {info.get_host()}{construct};\n',
        f'char_ptr = static_cast<char*>( static_cast<void*>({info.get_pinned()}) ) + n * {info.get_size(False)};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({use_ref}{info.get_host()}), {info.get_size(False)});\n\n',
    ])

def iterate_tilemetadata(connectors: dict, size_connectors: dict, tilemetadata: dict, language: str, num_arrays: int):
    """
    Iterates the tilemetadata section of the JSON.
    
    :param dict connectors: The dict containing all connectors for cgkit.
    :param dict size_connectors: The dict containing all size connectors for variable sizes.
    :param dict tilemetadata: The dict containing information from the tile-metadata section in the JSON.
    :param str language: The language to use
    :param int num_arrays: The number of arrays inside tile-in, tile-in-out, tile-out, and tile-scratch.
    """
    section_creation(jsc.T_MDATA, tilemetadata, connectors, size_connectors)
    connectors[_T_DESCRIPTOR] = []
    use_ref = ""
    memcpy_func = tmdata_memcpy_f if language == util.Language.fortran else cpp_helpers.tmdata_memcpy_cpp
    if language == util.Language.cpp: 
        cpp_helpers.insert_farray_size(size_connectors, num_arrays)
        use_ref = "&"
    
    for item,name in tilemetadata.items():
        try:
            item_type = util.TILE_VARIABLE_MAPPING[name] if language == util.Language.cpp else util.F_HOST_EQUIVALENT[util.TILE_VARIABLE_MAPPING[name]] 
        except Exception:
            warnings.warn(f"{name} was not found in tile_variable_mapping. Ignoring...")
            continue
        
        size_eq = f"MILHOJA_MDIM * sizeof({item_type})" if language == util.Language.fortran else f"sizeof({ util.FARRAY_MAPPING.get(item_type, item_type) })"
        info = dpinfo.DataPacketMemberVars(item=item, dtype=item_type, size_eq=size_eq, per_tile=True)

        connectors[_PUB_MEMBERS].extend( [f'{item_type}* {info.get_device()};\n'] )
        connectors[_SET_MEMBERS].extend( [f'{info.get_device()}{{nullptr}}'] )
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.get_size(False)} = {size_eq};\n' )
        set_pointer_determination(connectors, jsc.T_MDATA, info)

        info.dtype = util.FARRAY_MAPPING.get(item_type, item_type)
        connectors[_T_DESCRIPTOR].append(
            f'const auto {name} = tileDesc_h->{name}();\n'
        )

        info.dtype = info.dtype.replace('unsigned', '')
        if info.dtype in util.F_HOST_EQUIVALENT and language == util.Language.fortran:
            fix_index = '+1' if info.dtype == str('IntVect') else ''
            info.dtype = util.F_HOST_EQUIVALENT[info.dtype]
            construct_host = f"[MILHOJA_MDIM] = {{ {name}.I(){fix_index}, {name}.J(){fix_index}, {name}.K(){fix_index} }}"
            use_ref = ""
        else:
            construct_host = f' = static_cast<{item_type}>({info.get_host()})'
            use_ref = "&"

        memcpy_func(connectors=connectors, construct=construct_host, use_ref=use_ref, info=info, alt_name=name)
 
    # TODO: Remove this once bounds section is implemented in json. 
    missing_dependencies = cpp_helpers.get_metadata_dependencies(tilemetadata, language)
    connectors[_T_DESCRIPTOR].extend([
        f'const auto {item} = tileDesc_h->{item}();\n'
        for item in missing_dependencies
    ]) 

# TODO: This needs to be modified when converting from lbound to bound section.
# TODO TODO: This setup currently does not work, since lbound sizes may not always be 3 * sizeof(int) or IntVect.
def iterate_lbound(connectors: dict, size_connectors: dict, lbound: dict, lang: str):
    """
    Iterates the lbound section of the JSON.
    
    :param dict connectors: The dict containing all cgkit connectors.
    :param dict size_connectors: The dict containing all size connectors for items in the data packet.
    :param dict lbound: The dict containing the lbound section (This will likely be removed later).
    :param str lang: The language to use.
    """
    dtype = 'int' if lang == util.Language.fortran else 'IntVect'
    dtype_size = '3 * sizeof(int)' if lang == util.Language.fortran else 'IntVect'
    memcpy_func = tmdata_memcpy_f if lang == util.Language.fortran else cpp_helpers.tmdata_memcpy_cpp
    use_ref = '' if lang == util.Language.fortran else '&'
    lbound_mdata = ' + '.join( f'SIZE_{item.upper()}' for item in lbound ) if lbound else '0'
    size_connectors[f'size_{jsc.T_MDATA}'] = ' + '.join( [size_connectors[f'size_{jsc.T_MDATA}'], lbound_mdata])
    for key,bound in lbound.items():
        constructor_expression,memcpy_list = util.format_lbound_string(key, bound)
        info = dpinfo.DataPacketMemberVars(item=key, dtype=dtype, size_eq=f'sizeof({dtype_size})', per_tile=True)

        connectors[_PUB_MEMBERS].extend( [f'{dtype}* {info.get_device()};\n'] )
        connectors[_SET_MEMBERS].append( f'{info.get_device()}{{nullptr}}')
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n' )
        connectors[f'pointers_{jsc.T_MDATA}'].append(
            f"""{dtype}* {info.get_pinned()} = static_cast<{dtype}*>( static_cast<void*>(ptr_p) );\n""" + 
            f"""{info.get_device()} = static_cast<{dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_p += {info.get_size(True)};\n""" + 
            f"""ptr_d += {info.get_size(True)};\n\n"""
        )
        connectors[_T_DESCRIPTOR].append(
            f'const IntVect {key} = {constructor_expression};\n'
        )
        memcpy_func(connectors, f"[{len(memcpy_list)}] = {{{','.join(memcpy_list)}}}",
                    use_ref, info, '' )


def iterate_tilein(connectors: dict, size_connectors: dict, tilein: dict, _:dict, language: str) -> None:
    """
    Iterates the tile in section of the JSON.
    
    :param dict connectors: The dict containing all connectors for cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the data packet.
    :param dict tilein: The dict containing the information in the tile_in section.
    """
    del _
    section_creation(jsc.T_IN, tilein, connectors, size_connectors)
    pinnedLocation = set()
    for item,data in tilein.items():
        extents = data[jsc.EXTENTS]
        start = data[jsc.START]
        end = data[jsc.END]
        extents = ' * '.join(f'({item})' for item in extents)
        unks = f'{end} - {start} + 1'
        info = dpinfo.DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * ({unks}) * sizeof({data[jsc.DTYPE]})', per_tile=True)
        
        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.get_device()};\n'
            f'{info.dtype}* {info.get_pinned()};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [f'{info.get_device()}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        set_pointer_determination(connectors, jsc.T_IN, info, False)
        add_memcpy_connector(connectors, jsc.T_IN, extents, item, start, end, info.get_size(False), info.dtype)
        # temporary measure until the bounds information in JSON is solidified.
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, 'loGC', 'hiGC', unks, info.dtype)

    connectors[f'memcpy_{jsc.T_IN_OUT}'].extend(pinnedLocation)

def add_memcpy_connector(connectors: dict, section: str, extents: str, item: str, start: str, end: str, size_item: str, raw_type: str):
    """
    Adds a memcpy connector based on the information passed in.
    
    :param dict connectors: The dict containing all cgkit connectors
    :param str section: The section to add a memcpy connector for.
    :param str extents: The string containing array extents information.
    :param str item: The item to copy into pinned memory.
    :param str start: The starting index of the array.
    :param str end: The ending index of the array.
    :param str size_item: The string containing the size variable for *item*.
    :param str raw_type: The data type of the item.
    """
    connectors[f'memcpy_{section}'].extend([
        f'{raw_type}* {item}_d = tileDesc_h->dataPtr();\n'  # eventually we will pass arguments to data ptr for specific data args.
        f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
        f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});\n'
        f'char_ptr = static_cast<char*>( static_cast<void*>(_{item}_p) ) + n * {size_item};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({item}_d + offset_{item}), nBytes_{item});\n'
    ])

def add_unpack_connector(connectors: dict, section: str, extents, start, end, raw_type, in_ptr, out_ptr):
    """
    Adds an unpack connector to the connectors dictionary based on the information passed in.
    
    Parameters:
        connectors: dict - The connectors dictionary
        section: str - The name of the section
        extents: str - The extents of the array
        start: str - The start variable
        end: str - The end variable
        raw_type - The item's data type
        in_ptr: str - The name of the in data pointer
        out_ptr: str - The name of the out data pointer
    """
    connectors[_IN_PTRS].append(f'{raw_type}* {in_ptr}_data_h = tileDesc_h->dataPtr();\n')
    connectors[f'unpack_{section}'].extend([
        f'std::size_t offset_{in_ptr} = {extents} * static_cast<std::size_t>({start});\n',
        f'{raw_type}*        start_h_{in_ptr} = {in_ptr}_data_h + offset_{in_ptr};\n'
        f'const {raw_type}*  start_p_{out_ptr} = {out_ptr}_data_p + offset_{in_ptr};\n'
        f'std::size_t nBytes_{in_ptr} = {extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});\n',
        f'std::memcpy(static_cast<void*>(start_h_{in_ptr}), static_cast<const void*>(start_p_{out_ptr}), nBytes_{in_ptr});\n'
    ])
    connectors[_OUT_PTRS].append(f'{raw_type}* {out_ptr}_data_p = static_cast<{raw_type}*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _{out_ptr}_p ) ) + n * SIZE_{out_ptr.upper()} ) );\n')

def iterate_tileinout(connectors: dict, size_connectors: dict, tileinout: dict, _:dict, language: str) -> None:
    """
    Iterates the tileinout section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the JSON.
    :param dict tileinout: The dict containing the data from the tile-in-out section of the datapacket json.
    :param str language: The language to use.
    """
    section_creation(jsc.T_IN_OUT, tileinout, connectors, size_connectors)
    connectors[f'memcpy_{jsc.T_IN_OUT}'] = []
    connectors[f'unpack_{jsc.T_IN_OUT}'] = []
    pinnedLocation = set()
    for item,data in tileinout.items():
        start_in = data[jsc.START_IN]
        end_in = data[jsc.END_IN]
        start_out = data[jsc.START_OUT]
        end_out = data[jsc.END_OUT]
        extents = ' * '.join(f'({item})' for item in data[jsc.EXTENTS])
        unks = f'{end_in} - {start_in} + 1'
        info = dpinfo.DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * ({unks}) * sizeof({data[jsc.DTYPE]})', per_tile=True)

        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.get_device()};\n'
            f'{info.dtype}* {info.get_pinned()};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [f'{info.get_device()}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        set_pointer_determination(connectors, jsc.T_IN_OUT, info, False)
        add_memcpy_connector(connectors, jsc.T_IN_OUT, extents, item, start_in, end_in, info.get_size(False), info.dtype)
        add_unpack_connector(connectors, jsc.T_IN_OUT, extents, start_out, end_out, info.dtype, item, item)
        if language == util.Language.cpp:
            # hardcode lo and hi for now.
            cpp_helpers.insert_farray_memcpy(connectors, item, "loGC", "hiGC", unks, info.dtype)
    connectors[f'memcpy_{jsc.T_IN_OUT}'].extend(pinnedLocation)

def iterate_tileout(connectors: dict, size_connectors: dict, tileout: dict, _:dict, language: str) -> None:
    """
    Iterates the tileout section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the JSON.
    :param dict tileout: The dict containing information from the tile-out section of the data packet JSON.
    :param str language: The language to use. 
    """
    section_creation(jsc.T_OUT, tileout, connectors, size_connectors)
    connectors[f'unpack_{jsc.T_OUT}'] = []
    for item,data in tileout.items():
        start = data[jsc.START]
        end = data[jsc.END]
        extents = ' * '.join(f'({item})' for item in data[jsc.EXTENTS])
        info = dpinfo.DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * ( {end} - {start} + 1 ) * sizeof({data[jsc.DTYPE]})', per_tile=True)

        # TODO: An output array needs the key of the corresponding input array to know which array to pull from if multiple unks exist.
        corresponding_in_data = data.get('in_key', '') 

        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.get_device()};\n'
            f'{info.dtype}* {info.get_pinned()};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [f'{info.get_device()}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n'
        )
        set_pointer_determination(connectors, jsc.T_OUT, info, False)
        add_unpack_connector(connectors, jsc.T_OUT, extents, start, end, info.dtype, corresponding_in_data, info.ITEM)
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, cpp_helpers.BOUND_MAP[item][0], cpp_helpers.BOUND_MAP[item][1], cpp_helpers.BOUND_MAP[item][2], info.dtype)

def iterate_tilescratch(connectors: dict, size_connectors: dict, tilescratch: dict, language: str) -> None:
    """
    Iterates the tilescratch section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for variable sizes.
    :param dict tilescratch: The dict containing information from the tilescratch section of the JSON.
    :param str language: The language to use when generating the packet. 
    """
    section_creation(jsc.T_SCRATCH, tilescratch, connectors, size_connectors)
    for item,data in tilescratch.items():
        # lbound = f"lo{item[0].capitalize()}{item[1:]}"
        # hbound = ...
        extents = ' * '.join(f'({val})' for val in data[jsc.EXTENTS])
        info = dpinfo.DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * sizeof({data[jsc.DTYPE]})', per_tile=True)

        connectors[_PUB_MEMBERS].append( f'{info.dtype}* {info.get_device()};\n' )
        connectors[_SET_MEMBERS].append( f'{info.get_device()}{{nullptr}}' )
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.get_size(False)} = {info.SIZE_EQ};\n' )
        connectors[f'pointers_{jsc.T_SCRATCH}'].append(
            f"""{info.get_device()} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_d += {info.get_size(info.PER_TILE)};\n\n"""
        )
        
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, cpp_helpers.BOUND_MAP[item][0], cpp_helpers.BOUND_MAP[item][1], cpp_helpers.BOUND_MAP[item][2], info.dtype)

def sort_dict(section, sort_key) -> dict:
    """
    Sorts a given dictionary using the sort key.
    
    :param dict section: The dictionary to sort.
    :param func sort_key: The function to sort with.
    """
    return dict( sorted(section, key = sort_key, reverse = True) )

def write_connectors(template, connectors: dict):
    """
    Writes connectors to the datapacket file.
    
    :param template: The file to write the connectors to.
    :param dict connectors: The dict containing all cgkit connectors.
    """
    # constructor args requires a special formatting 
    template.writelines([
        f'/* _connector:constructor_args */\n'
        ] + [ ','.join(connectors[_CON_ARGS]) ] + ['\n\n'])
    del connectors[_CON_ARGS]

    # set members needs a special formatting
    template.writelines(
        [ f'/* _connector:set_members */\n'] + 
        [',\n'.join(connectors[_SET_MEMBERS]) ] + 
        ['\n\n']
    )
    del connectors[_SET_MEMBERS]

    template.writelines(
        [ f'/* _connector:host_members */\n' ] + 
        [','.join(connectors[_HOST_MEMBERS])] + 
        ['\n\n']
    )
    del connectors[_HOST_MEMBERS]

    # write any leftover connectors
    for connection in connectors:
        template.writelines([
            f'/* _connector:{connection} */\n'
        ] + [ ''.join(connectors[connection]) ] + ['\n'] 
    )

# Not really necessary to use constant string keys for this function
# since param keys do not get used outside of this function.
def set_default_params(data: dict, params: dict):
    """
    Sets the default parameters for cgkit.
    
    :param dict data: The dict containing the data packet JSON data.
    :param dict params: The dict containing all parameters for the outer template.
    """
    try:
        params['align_size'] = int(data.get(jsc.BYTE_ALIGN, 16))
        params['n_extra_streams'] = int(data.get(jsc.EXTRA_STREAMS, 0))
        params['class_name'] = data[jsc.NAME]
        params['ndef_name'] = f'{data[jsc.NAME].upper()}_UNIQUE_IFNDEF_H_'
    except:
        print("align_size or n_extra_streams is not an integer.")
    
def generate_outer(name: str, params: dict):
    """
    Generates the outer template for the datapacket template.
    
    :param str name: The name of the class.
    :param dict params: The dict containing the parameter list to write to the outer template.
    """
    with open(name, 'w') as outer:
        outer.writelines(
        [
            '/* _connector:datapacket_outer */\n',
            '/* _link:datapacket */\n'
        ] + 
        ['\n'.join(f'/* _param:{item} = {params[item]} */' for item in params)]
    )

def write_size_connectors(size_connectors: dict, file):
    """
    Writes the size connectors to the specified file.

    :param dict size_connectors: The dictionary of size connectors for use with CGKit.
    :param TextIO file: The file to write to.
    :rtype: None
    """
    for key,item in size_connectors.items():
        file.write(f'/* _connector:{key} */\n{item}\n\n')
        
#TODO: There is a large issue with size sorting. For array types, sizes are determined by sizes as well as
#      the given extents and unk vars. Since the extents are passed in as mathematical expressions, we would
#      need to write an expression parser in order to get accurate size sorting. Either that, or the expressions
#      given in the JSON file need to be single constants and not mathematical expressions
def generate_helper_template(data: dict) -> None:
    """
    Generates the helper template with the provided JSON data.
    
    :param dict data: The dictionary containing the DataPacket JSON data.
    """
    with open(data[jsc.HELPERS], 'w') as template:
        size_connectors = defaultdict(str)
        connectors = defaultdict(list) 
        params = defaultdict(str)
        lang = data[jsc.LANG]

        # # Read every section in the JSON.

        # set defaults for the connectors. 
        connectors[_CON_ARGS] = []
        connectors[_SET_MEMBERS] = []
        connectors[_SIZE_DET] = []
        set_default_params(data, params)

        sort_func = lambda key_and_type: sizes.get(key_and_type[1], 0) if sizes else 1
        sizes = data[jsc.SIZES]
        tpv = data.get(jsc.GENERAL, {}) # tpv = thread-private-variables
        if 'nTiles' in tpv:
            warnings.warn("Found nTiles in data packet. Mistake?")
        tpv['nTiles'] = 'int' # every data packet always needs nTiles so we insert it here.
        tpv = tpv.items()
        iterate_constructor(connectors, size_connectors, sort_dict(tpv, sort_func))

        num_arrays = len( data.get(jsc.T_SCRATCH, {}) ) + len(data.get(jsc.T_IN, {})) + \
                     len(data.get(jsc.T_IN_OUT, {})) + len(data.get(jsc.T_OUT, {}))
        if lang == util.Language.cpp:
            sort_func = lambda x: sizes.get(util.TILE_VARIABLE_MAPPING[x[1]], 0) if sizes else 1
        elif lang == util.Language.fortran:
            sort_func = lambda x: sizes.get(util.F_HOST_EQUIVALENT[util.TILE_VARIABLE_MAPPING[x[1]]], 0) if sizes else 1
        # sort_func = lambda x: sizes.get(util.TILE_VARIABLE_MAPPING[x[1]], 0) if sizes else 1
        metadata = data.get(jsc.T_MDATA, {}).items()
        iterate_tilemetadata(connectors, size_connectors, sort_dict(metadata, sort_func), lang, num_arrays)

        lbound = data.get(jsc.LBOUND, {})
        iterate_lbound(connectors, size_connectors, lbound, lang)

        sort_func = lambda x: sizes.get(x[1][jsc.DTYPE], 0) if sizes else 1
        for section,funct in {jsc.T_IN: iterate_tilein, jsc.T_IN_OUT: iterate_tileinout,
                              jsc.T_OUT: iterate_tileout }.items():
            dictionary = data.get(section, {}).items()
            funct(connectors, size_connectors, sort_dict(dictionary, sort_func), params, lang)
        tilescratch = data.get(jsc.T_SCRATCH, {}).items()
        iterate_tilescratch(connectors, size_connectors, sort_dict(tilescratch, sort_func), lang)

        if lang == util.Language.cpp: 
            cpp_helpers.insert_farray_information(data, connectors, _PUB_MEMBERS)

        # Write to all files.
        generate_outer(data[jsc.OUTER], params)
        write_size_connectors(size_connectors, template)
        generate_extra_streams_information(connectors, data.get(jsc.EXTRA_STREAMS, 0))
        write_connectors(template, connectors)

        
