"""
This is the main script for generating a new DataPacket. The overall structure of this 
code is to move through every possible DataPacket JSON section and fill out the various CGKkit dictionaries
for generating every template.

TODO: How to sort bound class members by size?
TODO: Eventually logging should be more informative and replace all print statements.
"""

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
# import DataPacketMemberVars as dpinfo
from DataPacketMemberVars import DataPacketMemberVars

def _add_size_parameter(name: str, section_dict: dict, connectors: dict):
    """
    Adds a size connector to the params dict that is passed in. The size connector dictionary stores every size 
    of every item in the data packet for use with a CGKit template.

    :param str name: The name of the size connector. Usually the name of a section.
    :param dict section_dict: The section used to generate sizes. Contains all of the items in the section.
    :param dict connectors: The dictionary containing all connectors generated for the packet.
    :rtype: None
    """
    connectors[f'size_{name}'] = ' + '.join( f'SIZE_{item.upper()}' for item in section_dict) if section_dict else '0'

def _section_creation(name: str, section: dict, connectors: dict, size_connectors):
    """
    Creates a section and sets the default value to an empty list. It's assumed that the function that calls this method 
    will populate the dictionary using the same name.
    
    :param str name: The name of the section to create.
    :param dict section: The dictionary to get all data packet items from. 
    :param dict connectors: The dictionary containing all link connectors. 
    :param dict size_connectors: The dictionary containing all connectors that determine sizes for each variable in the data packet.
    """
    _add_size_parameter(name, section, size_connectors)
    connectors[f'pointers_{name}'] = []
    connectors[f'memcpy_{name}'] = []

def _set_pointer_determination(connectors: dict, section: str, info: DataPacketMemberVars, item_is_member_variable=True):
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
    
    # insert items into boiler plate for the pointer determination phase for *section*.
    connectors[f'pointers_{section}'].append(
        f"""{dtype}{info.pinned} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_p) );\n""" + 
        f"""{info.device} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
        f"""ptr_p+={info.total_size};\n""" + 
        f"""ptr_d+={info.total_size};\n\n"""
    )

def _generate_extra_streams_information(connectors: dict, extra_streams: int):
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

    # these insert the various stream functions if there are more than 1 stream
    # normally these would be in the template but these functions have no reason to exist if the
    # task function does not use more than 1 stream.
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

    # Inserts the code necessary to acquire extra streams. 
    connectors[jsc.EXTRA_STREAMS].extend([
        f'stream{i}_ = RuntimeBackend::instance().requestStream(true);\n' + 
        f'if(!stream{i}_.isValid()) throw std::runtime_error("[_param:class_name::pack] Unable to acquire second stream");\n'
        for i in range(2, extra_streams+2)
    ])

def _iterate_constructor(connectors: dict, size_connectors: dict, constructor: dict):
    """
    Iterates the constructor / thread-private-variables section and adds the necessary connectors.

    :param dict connectors: The dictionary containing all connectors for use with CGKit.
    :param dict size_connectors: The dictionary containing all size connectors for determining the sizes of each item in the packet.
    :param dict constructor: The dictionary containing the data for the ptv section in the DataPacket JSON.
    :rtype: None
    """
    _section_creation(jsc.GENERAL, constructor, connectors, size_connectors)
    connectors[_HOST_MEMBERS] = []
    # # MOVE THROUGH EVERY CONSTRUCTOR ITEM
    for key,item_type in constructor.items():
        info = DataPacketMemberVars(item=key, dtype=item_type, size_eq=f'sizeof({item_type})', per_tile=False)

        # nTiles is a special case here. nTiles should not be included in the constructor, and it has its own host variable generation.
        if key != 'nTiles':
            connectors[_CON_ARGS].append( f'{item_type} {key}' )
            connectors[_HOST_MEMBERS].append( info.host )
        # add the necessary connectors for the constructor section.
        connectors[_PUB_MEMBERS].extend(
            [f'{info.dtype} {info.host};\n', 
             f'{info.dtype}* {info.device};\n']
        )
        set_host = "{ tiles_.size() }" if key == "nTiles" else f"{{ {key} }}"
        connectors[_SET_MEMBERS].extend(
            [f'{info.host}{set_host}', 
             f'{info.device}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        _set_pointer_determination(connectors, jsc.GENERAL, info)
        connectors[f'memcpy_{jsc.GENERAL}'].append(
            f'std::memcpy({info.pinned}, static_cast<void*>(&{info.host}), {info.size});\n'
        )

def _tmdata_memcpy_f(connectors: dict, construct: str, use_ref: str, info: DataPacketMemberVars):
    """
    Adds the memcpy portion for the metadata section in a fortran packet.
    
    :param dict connectors: The dictionary containing all cgkit connectors.
    :param str construct: The generated host variable to copy to the pinned pointer location
    :param str use_ref: Use a reference to the host item.
    :param DataPacketMemberVars info: Contains information for formatting the name to get variable names.
    :param str alt_name: Unused
    """
    connectors[f'memcpy_{jsc.T_MDATA}'].extend([
        f'{info.dtype} {info.host}{construct};\n',
        f'char_ptr = static_cast<char*>( static_cast<void*>({info.pinned}) ) + n * {info.size};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({use_ref}{info.host}), {info.size});\n\n',
    ])

def _tmetadata_memcopy(connectors: dict, construct: str, use_ref: str, info: DataPacketMemberVars, alt_name: str, language: str):
    """
    :param dict connectors: The dictionary containing all connectors for use with cgkit.
    :param str construct: The variable name containing the value to be copied using memcpy.
    :param str use_ref: string that determines if a variable should be passed by reference or not.
    :param dpinfo.DataPacketMemberVars
    """
    if language == util.Language.cpp:
        cpp_helpers.tmdata_memcpy_cpp(connectors, info, alt_name)
    elif language == util.Language.fortran:
        _tmdata_memcpy_f(connectors, construct, use_ref, info)
    else:
        print("Incompatible language.")

def _iterate_tilemetadata(connectors: dict, size_connectors: dict, tilemetadata: dict, language: str, num_arrays: int):
    """
    Iterates the tilemetadata section of the JSON.
    
    :param dict connectors: The dict containing all connectors for cgkit.
    :param dict size_connectors: The dict containing all size connectors for variable sizes.
    :param dict tilemetadata: The dict containing information from the tile-metadata section in the JSON.
    :param str language: The language to use
    :param int num_arrays: The number of arrays inside tile-in, tile-in-out, tile-out, and tile-scratch.
    """
    _section_creation(jsc.T_MDATA, tilemetadata, connectors, size_connectors)
    connectors[_T_DESCRIPTOR] = []
    if language == util.Language.cpp: 
        cpp_helpers.insert_farray_size(size_connectors, num_arrays)
    
    for item,name in tilemetadata.items():
        try:
            item_type = util.TILE_VARIABLE_MAPPING[name] if language == util.Language.cpp else util.F_HOST_EQUIVALENT[util.TILE_VARIABLE_MAPPING[name]] 
        except Exception:
            # TODO: Eventually use the logger to track this.
            warnings.warn(f"{name} was not found in tile_variable_mapping. Ignoring...")
            continue
        
        # code generation for the tile_metadata section often depends on the language to properly generate
        # working code
        size_eq = f"MILHOJA_MDIM * sizeof({item_type})" if language == util.Language.fortran else f"sizeof({ util.FARRAY_MAPPING.get(item_type, item_type) })"
        info = DataPacketMemberVars(item=item, dtype=item_type, size_eq=size_eq, per_tile=True)

        # extend each connector
        connectors[_PUB_MEMBERS].extend( [f'{item_type}* {info.device};\n'] )
        connectors[_SET_MEMBERS].extend( [f'{info.device}{{nullptr}}'] )
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.size} = {size_eq};\n' )
        _set_pointer_determination(connectors, jsc.T_MDATA, info)

        # data type depends on the language. If there exists a mapping for a fortran data type for the given item_type,
        # use that instead.
        info.dtype = util.FARRAY_MAPPING.get(item_type, item_type)
        connectors[_T_DESCRIPTOR].append(
            f'const auto {name} = tileDesc_h->{name}();\n'
        )

        # unsigned does not matter in fortran.
        # TODO: This might not work completely when needing an unsigned int 
        #       as a variable type. Try generating spark packets to resolve this.
        info.dtype = info.dtype.replace('unsigned', '')
        # if the language is fortran and there exists a fortran data type equivalent (eg, IntVect -> int array.)
        use_ref = ""
        if info.dtype in util.F_HOST_EQUIVALENT and language == util.Language.fortran:
            fix_index = '+1' if info.dtype == str('IntVect') else '' # indices are 1 based, so bound arrays need to adjust
            info.dtype = util.F_HOST_EQUIVALENT[info.dtype]
            construct_host = f"[MILHOJA_MDIM] = {{ {name}.I(){fix_index}, {name}.J(){fix_index}, {name}.K(){fix_index} }}"
            use_ref = "" # don't need to pass by reference with primitive arrays
        else:
            construct_host = f' = static_cast<{item_type}>({info.host})'
            use_ref = "&" # need a reference for Vect objects.

        _tmetadata_memcopy(connectors, construct_host, use_ref, info, name, language)
 
    # TODO: Remove this once bounds information is implemented.
    missing_dependencies = cpp_helpers.get_metadata_dependencies(tilemetadata, language)
    connectors[_T_DESCRIPTOR].extend([
        f'const auto {item} = tileDesc_h->{item}();\n'
        for item in missing_dependencies
    ]) 

# TODO: This needs to be modified when converting from lbound to bound section.
# TODO TODO: This setup currently does not work, since lbound sizes may not always be 3 * sizeof(int) or IntVect.
# This function is kind of confusion and I'm intentionally ignoring it for now,
# it depends on how bounds is implemented in the JSON. Bits and pieces of this might end up existing
# within the tile data / array sections.
# Luckily, this function is relatively straightforward. 
def _iterate_lbound(connectors: dict, size_connectors: dict, lbound: dict, lang: str):
    """
    Iterates the lbound section of the JSON.
    
    :param dict connectors: The dict containing all cgkit connectors.
    :param dict size_connectors: The dict containing all size connectors for items in the data packet.
    :param dict lbound: The dict containing the lbound section (This will likely be removed later).
    :param str lang: The language to use.
    """
    dtype = 'int' if lang == util.Language.fortran else 'IntVect'
    dtype_size = '3 * sizeof(int)' if lang == util.Language.fortran else 'IntVect'
    use_ref = '' if lang == util.Language.fortran else '&'
    lbound_mdata = ' + '.join( f'SIZE_{item.upper()}' for item in lbound ) if lbound else '0'
    size_connectors[f'size_{jsc.T_MDATA}'] = ' + '.join( [size_connectors[f'size_{jsc.T_MDATA}'], lbound_mdata])
    for key,bound in lbound.items():
        constructor_expression,memcpy_list = util.format_lbound_string(key, bound)
        info = DataPacketMemberVars(item=key, dtype=dtype, size_eq=f'sizeof({dtype_size})', per_tile=True)

        connectors[_PUB_MEMBERS].extend( [f'{dtype}* {info.device};\n'] )
        connectors[_SET_MEMBERS].append( f'{info.device}{{nullptr}}')
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n' )
        connectors[f'pointers_{jsc.T_MDATA}'].append(
            f"""{dtype}* {info.pinned} = static_cast<{dtype}*>( static_cast<void*>(ptr_p) );\n""" + 
            f"""{info.device} = static_cast<{dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_p += {info.total_size};\n""" + 
            f"""ptr_d += {info.total_size};\n\n"""
        )
        connectors[_T_DESCRIPTOR].append(
            f'const IntVect {key} = {constructor_expression};\n'
        )
        _tmetadata_memcopy(connectors, f"[{len(memcpy_list)}] = {{{','.join(memcpy_list)}}}", use_ref, info, '', lang)


def _iterate_tilein(connectors: dict, size_connectors: dict, tilein: dict, language: str) -> None:
    """
    Iterates the tile in section of the JSON.
    
    :param dict connectors: The dict containing all connectors for cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the data packet.
    :param dict tilein: The dict containing the information in the tile_in section.
    :param str language: The language of the corresponding task function.
    """
    _section_creation(jsc.T_IN, tilein, connectors, size_connectors)
    pinnedLocation = set()
    for item,data in tilein.items():
        # gather all information from tile_in section.
        extents = data[jsc.EXTENTS]
        start = data.get(jsc.START, "")
        end = data.get(jsc.END, "")
        extents = ' * '.join(f'({item})' for item in extents)
        unks = f'{end if end else 0} - {start if start else 0} + 1'
        info = DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * ({unks}) * sizeof({data[jsc.DTYPE]})', per_tile=True)
        
        # Add necessary connectors.
        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.device};\n'
            f'{info.dtype}* {info.pinned};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [
                f'{info.device}{{nullptr}}',
                f'{info.pinned}{{nullptr}}'
            ]
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        _set_pointer_determination(connectors, jsc.T_IN, info, False)
        _add_memcpy_connector(connectors, jsc.T_IN, extents, item, start, end, info.size, info.dtype)
        # temporary measure until the bounds information in JSON is solidified.
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, 'loGC', 'hiGC', unks, info.dtype)

    connectors[f'memcpy_{jsc.T_IN_OUT}'].extend(pinnedLocation)


def get_data_pointer_string(item: str):
    # todo: get data from structure_index section
    data_pointer_string = "tileDesc_h->dataPtr()"
    # TODO: Make this actually good.    
    # TODO: THis is testing for getting flux data pointers.
    # TODO: Something like this would probably be a 'per format' thing and not a data packet generator specification.
    if 'flux' in item.lower() or 'fl' in item.lower():
        if 'X' in item: data_pointer_string = util.DATA_POINTER_MAP['flX']
        elif 'Y' in item: data_pointer_string = util.DATA_POINTER_MAP['flY']
        elif 'Z' in item: data_pointer_string = util.DATA_POINTER_MAP['flZ']
    return data_pointer_string


def _add_memcpy_connector(connectors: dict, section: str, extents: str, item: str, start: int, end: int, size_item: str, raw_type: str):
    """
    Adds a memcpy connector based on the information passed in.
    
    :param dict connectors: The dict containing all cgkit connectors
    :param str section: The section to add a memcpy connector for.
    :param str extents: The string containing array extents information.
    :param str item: The item to copy into pinned memory.
    :param int start: The starting index of the array.
    :param int end: The ending index of the array.
    :param str size_item: The string containing the size variable for *item*.
    :param str raw_type: The data type of the item.
    """

    # Developer's Note:
    # I'm adding the potential for not including a start and an ending index
    # to satisfy a small requirement by Anshu's requested DataPacket JSON 
    # generator. Ideally, bounds should always be specified in the 
    # JSON, and it's up to the application, not the DataPacket Generator, 
    # to specify defaults. This however, I think is an okay workaround for 
    # having default starting and ending indices, since copying the size of 
    # the entire variable is not application specific, and the size of the 
    # variable must be specified by the application (aka, no default sizes).
    offset = f"{extents} * static_cast<std::size_t>({start})"
    nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type})'
    if not start or not end:
        if not start: print("WARNING: You are missing a starting index. If this is intentional, ignore this warning.")
        if not end: print("WARNING: You are missing an ending index. If this is intentional, ignore this warning.")
        offset = '0'
        nBytes = size_item

    # This exists inside the pack function for copying data from tile_in to the device.
    # Luckily we don't really need to use DataPacketMemberVars here because the temporary device pointer 
    # is locally scoped.

    data_pointer_string = get_data_pointer_string(item)

    # TODO: Use grid data to get data pointer information.
    connectors[f'memcpy_{section}'].extend([
        f'{raw_type}* {item}_d = {data_pointer_string};\n'  # eventually we will pass arguments to data ptr for specific data args.
        f'constexpr std::size_t offset_{item} = {offset};\n',
        f'constexpr std::size_t nBytes_{item} = {nBytes};\n',
        f'char_ptr = static_cast<char*>( static_cast<void*>(_{item}_p) ) + n * {size_item};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({item}_d + offset_{item}), nBytes_{item});\n\n'
    ])

def _add_unpack_connector(connectors: dict, section: str, extents, start: int, end: int, raw_type: str, in_ptr: str, out_ptr: str):
    """
    Adds an unpack connector to the connectors dictionary based on the information passed in.
    
    :param dict connectors: The connectors dictionary
    :param str section: The name of the section
    :param str extents: The extents of the array
    :param int start: The start variable
    :param int end: The end variable
    :param str raw_type: The item's data type
    :param str in_ptr: The name of the in data pointer
    :param str out_ptr: The name of the out data pointer
    """

    # Developer's Note:
    # I'm adding the potential for not including a start and an ending index
    # to satisfy a small requirement by Anshu's requested DataPacket JSON 
    # generator. Ideally, bounds should always be specified in the 
    # JSON, and it's up to the application, not the DataPacket Generator, 
    # to specify defaults. This however, I think is an okay workaround for 
    # having default starting and ending indices, since copying the size of 
    # the entire variable is not application specific, and the size of the 
    # variable must be specified by the application (aka, no default sizes).
    offset = f"{extents} * static_cast<std::size_t>({start});"
    nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});'
    if not start or not end:
        if not start: print("WARNING: You are missing a starting index. If this is intentional, ignore this warning.")
        if not end: print("WARNING: You are missing an ending index. If this is intentional, ignore this warning.")
        offset = '0;'
        nBytes = f'SIZE_{out_ptr.upper()};'

    data_pointer_string = get_data_pointer_string(in_ptr)

    # TODO: Eventually the tile wrapper class will allow us to pick out the exact data array we need with dataPtr().
    connectors[_IN_PTRS].append(f'{raw_type}* {in_ptr}_data_h = {data_pointer_string};\n')
    connectors[f'unpack_{section}'].extend([
        f'constexpr std::size_t offset_{in_ptr} = {offset}\n',
        f'{raw_type}*        start_h_{in_ptr} = {in_ptr}_data_h + offset_{in_ptr};\n'
        f'const {raw_type}*  start_p_{out_ptr} = {out_ptr}_data_p + offset_{in_ptr};\n'
        f'constexpr std::size_t nBytes_{out_ptr} = {nBytes}\n',
        f'std::memcpy(static_cast<void*>(start_h_{in_ptr}), static_cast<const void*>(start_p_{out_ptr}), nBytes_{out_ptr});\n\n'
    ])
    # I'm the casting here is awful but I'm not sure there's a way around it that isn't just using c-style casting, 
    # and that is arguably worse than CPP style casting
    connectors[_OUT_PTRS].append(f'{raw_type}* {out_ptr}_data_p = static_cast<{raw_type}*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _{out_ptr}_p ) ) + n * SIZE_{out_ptr.upper()} ) );\n')

def _iterate_tileinout(connectors: dict, size_connectors: dict, tileinout: dict, language: str) -> None:
    """
    Iterates the tileinout section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the JSON.
    :param dict tileinout: The dict containing the data from the tile-in-out section of the datapacket json.
    :param str language: The language to use.
    """
    _section_creation(jsc.T_IN_OUT, tileinout, connectors, size_connectors)
    connectors[f'memcpy_{jsc.T_IN_OUT}'] = []
    connectors[f'unpack_{jsc.T_IN_OUT}'] = []
    pinnedLocation = set()
    # unpack all items in tile_in_out
    for item,data in tileinout.items():
        start_in = data.get(jsc.START_IN, "")
        end_in = data.get(jsc.END_IN, "")
        start_out = data.get(jsc.START_OUT, "")
        end_out = data.get(jsc.END_OUT, "")
        extents = ' * '.join(f'({item})' for item in data[jsc.EXTENTS])
        unks = f'{end_in if end_in else 0} - {start_in if start_in else 0} + 1'
        info = DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * ({unks}) * sizeof({data[jsc.DTYPE]})', per_tile=True)

        # set connectors
        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.device};\n'
            f'{info.dtype}* {info.pinned};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [
                f'{info.device}{{nullptr}}',
                f'{info.pinned}{{nullptr}}'
            ]
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        _set_pointer_determination(connectors, jsc.T_IN_OUT, info, False)
        _add_memcpy_connector(connectors, jsc.T_IN_OUT, extents, item, start_in, end_in, info.size, info.dtype)
        # here we pass in item twice because tile_in_out pointers get packed and unpacked from the same location.
        _add_unpack_connector(connectors, jsc.T_IN_OUT, extents, start_out, end_out, info.dtype, item, item)
        if language == util.Language.cpp:
            # hardcode lo and hi for now.
            cpp_helpers.insert_farray_memcpy(connectors, item, "loGC", "hiGC", unks, info.dtype)
    connectors[f'memcpy_{jsc.T_IN_OUT}'].extend(pinnedLocation)

def _iterate_tileout(connectors: dict, size_connectors: dict, tileout: dict, language: str) -> None:
    """
    Iterates the tileout section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for items in the JSON.
    :param dict tileout: The dict containing information from the tile-out section of the data packet JSON.
    :param str language: The language to use. 
    """
    _section_creation(jsc.T_OUT, tileout, connectors, size_connectors)
    connectors[f'unpack_{jsc.T_OUT}'] = []
    for item,data in tileout.items():
        # ge tile_out information
        start = data.get(jsc.START, "")
        end = data.get(jsc.END, "")
        extents = ' * '.join(f'({item})' for item in data[jsc.EXTENTS])
        info = DataPacketMemberVars(
            item=item, 
            dtype=data[jsc.DTYPE], 
            size_eq=f'{extents} * ( {end if end else 0} - {start if start else 0} + 1 ) * sizeof({data[jsc.DTYPE]})', 
            per_tile=True
        )

        # TODO: An output array needs the key of the corresponding input array to know which array to pull from if multiple unks exist.
        #       This may not need to exist once the generated tile wrapper class is created.
        corresponding_in_data = data.get('in_key', '') 

        connectors[_PUB_MEMBERS].append(
            f'{info.dtype}* {info.device};\n'
            f'{info.dtype}* {info.pinned};\n'
        )
        connectors[_SET_MEMBERS].extend(
            [f'{info.device}{{nullptr}}']
        )
        connectors[_SIZE_DET].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        connectors[_PINNED_SIZES].append(
            f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n'
        )
        _set_pointer_determination(connectors, jsc.T_OUT, info, False)
        _add_unpack_connector(connectors, jsc.T_OUT, extents, start, end, info.dtype, corresponding_in_data, info.ITEM)
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, cpp_helpers.BOUND_MAP[item][0], cpp_helpers.BOUND_MAP[item][1], f'{end} - {start} + 1', info.dtype)

def _iterate_tilescratch(connectors: dict, size_connectors: dict, tilescratch: dict, language: str) -> None:
    """
    Iterates the tilescratch section of the JSON.
    
    :param dict connectors: The dict containing all connectors for use with cgkit.
    :param dict size_connectors: The dict containing all size connectors for variable sizes.
    :param dict tilescratch: The dict containing information from the tilescratch section of the JSON.
    :param str language: The language to use when generating the packet. 
    """
    _section_creation(jsc.T_SCRATCH, tilescratch, connectors, size_connectors)
    for item,data in tilescratch.items():
        # lbound = f"lo{item[0].capitalize()}{item[1:]}"
        # hbound = ...
        extents = ' * '.join(f'({val})' for val in data[jsc.EXTENTS])
        info = DataPacketMemberVars(item=item, dtype=data[jsc.DTYPE], size_eq=f'{extents} * sizeof({data[jsc.DTYPE]})', per_tile=True)

        connectors[_PUB_MEMBERS].append( f'{info.dtype}* {info.device};\n' )
        connectors[_SET_MEMBERS].append( f'{info.device}{{nullptr}}' )
        connectors[_SIZE_DET].append( f'constexpr std::size_t {info.size} = {info.SIZE_EQ};\n' )
        connectors[f'pointers_{jsc.T_SCRATCH}'].append(
            f"""{info.device} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_d += {info.total_size};\n\n"""
        )
        # we don't insert into memcpy or unpack because the scratch is only used in the device memory 
        # and does not come back.
        
        # TODO: How to insert this FArray into the C++ packet? 
        #       We do not have control over the number of unknowns in scratch arrays, so how do we 
        #       incorporate this with lbound?
        if language == util.Language.cpp:
            cpp_helpers.insert_farray_memcpy(connectors, item, cpp_helpers.BOUND_MAP[item][0], cpp_helpers.BOUND_MAP[item][1], cpp_helpers.BOUND_MAP[item][2], info.dtype)

def _sort_dict(section, sort_key) -> dict:
    """
    Sorts a given dictionary using the sort key.
    
    :param dict section: The dictionary to sort.
    :param func sort_key: The function to sort with.
    """
    return dict( sorted(section, key = sort_key, reverse = True) )

def _write_connectors(template, connectors: dict):
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
def _set_default_params(data: dict, params: dict):
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
    
def _generate_outer(name: str, params: dict):
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

def _write_size_connectors(size_connectors: dict, file):
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
#      given in the JSON file need to be single constants and not mathematical expressions.
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
        _set_default_params(data, params)

        # # SETUP FOR CONSTRUCTOR  
        sort_func = lambda key_and_type: sizes.get(key_and_type[1], 0) if sizes else 1
        sizes = data[jsc.SIZES]
        tpv = data.get(jsc.GENERAL, {}) # tpv = thread-private-variables
        if 'nTiles' in tpv:
            warnings.warn("Found nTiles in data packet. Mistake?")
        tpv['nTiles'] = 'int' # every data packet always needs nTiles so we insert it here.
        tpv = tpv.items()
        _iterate_constructor(connectors, size_connectors, _sort_dict(tpv, sort_func))

        # # SETUP FOR TILE_METADATA
        num_arrays = len( data.get(jsc.T_SCRATCH, {}) ) + len(data.get(jsc.T_IN, {})) + \
                     len(data.get(jsc.T_IN_OUT, {})) + len(data.get(jsc.T_OUT, {}))
        if lang == util.Language.cpp:
            sort_func = lambda x: sizes.get(util.TILE_VARIABLE_MAPPING[x[1]], 0) if sizes else 1
        elif lang == util.Language.fortran:
            sort_func = lambda x: sizes.get(util.F_HOST_EQUIVALENT[util.TILE_VARIABLE_MAPPING[x[1]]], 0) if sizes else 1
        # sort_func = lambda x: sizes.get(util.TILE_VARIABLE_MAPPING[x[1]], 0) if sizes else 1
        metadata = data.get(jsc.T_MDATA, {}).items()
        _iterate_tilemetadata(connectors, size_connectors, _sort_dict(metadata, sort_func), lang, num_arrays)

        # # SETUP FOR LBOUND, TODO: Maybe this won't exist anymore
        lbound = data.get(jsc.LBOUND, {})
        _iterate_lbound(connectors, size_connectors, lbound, lang)

        # # SETUP EVERY ARRAY SECTION
        sort_func = lambda x: sizes.get(x[1][jsc.DTYPE], 0) if sizes else 1
        for section,funct in {jsc.T_IN: _iterate_tilein, jsc.T_IN_OUT: _iterate_tileinout,
                              jsc.T_OUT: _iterate_tileout }.items():
            dictionary = data.get(section, {}).items()
            funct(connectors, size_connectors, _sort_dict(dictionary, sort_func), lang)
        tilescratch = data.get(jsc.T_SCRATCH, {}).items()
        _iterate_tilescratch(connectors, size_connectors, _sort_dict(tilescratch, sort_func), lang)

        # insert farray variables if necessary.
        if lang == util.Language.cpp: 
            cpp_helpers.insert_farray_information(data, connectors, _PUB_MEMBERS, _SET_MEMBERS)

        # Write to all files.
        _generate_outer(data[jsc.OUTER], params)
        _write_size_connectors(size_connectors, template)
        _generate_extra_streams_information(connectors, data.get(jsc.EXTRA_STREAMS, 0))
        _write_connectors(template, connectors)

        
