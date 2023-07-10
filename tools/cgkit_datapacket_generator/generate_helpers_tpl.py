from collections import defaultdict
import utility as consts
import warnings
import cpp_helpers
import json_sections

def add_size_parameter(name: str, section_dict: dict, connectors: dict):
    """
    Adds a size connector to the params dict that is passed in.

    Parameters:
        name: name of the size connector.\n
        section_dict: the section to use to generate sizes.\n
        connectors: the dictionary to inserttile_descriptor into.
    Returns:
        None
    """
    connectors[f'size_{name.replace("-", "")}'] = ' + '.join( f'SIZE_{item.upper()}' for item in section_dict) if section_dict else 0

def section_creation(name: str, section: dict, connectors: dict, size_connectors):
    add_size_parameter(name, section, size_connectors)
    connectors[f'pointers_{name}'] = []
    connectors[f'memcpy_{name}'] = []

def set_pointer_determination(connectors: dict, section: str, item:str, device_item: str, size_item: str, item_type, use_item_type=True):
    dtype = item_type
    if not use_item_type:
        dtype = f""
    else:
        dtype += "*"
    connectors[f'pointers_{section}'].append(
        f"""{dtype} {item}_p = static_cast<{item_type}*>( static_cast<void*>(ptr_p) );\n""" + 
        f"""{device_item} = static_cast<{item_type}*>( static_cast<void*>(ptr_d) );\n""" + 
        f"""ptr_p+={size_item};\n""" + 
        f"""ptr_d+={size_item};\n\n"""
    )

def generate_extra_streams_information(connectors: dict, extra_streams: int):
    """
    Fills the links extra_streams, stream_functions_h/cxx.

    Parameters:
        connectors: Connectors dictionary\n
        extra_streams: The number of extra streams

    Returns:
        None
    """
    if extra_streams < 1: return

    connectors['stream_functions_h'].extend([
        f'int extraAsynchronousQueue(const unsigned int id) override;\n',
        f'void releaseExtraQueue(const unsigned int id) override;\n'
    ])
    connectors['extra_streams'].extend([
        f'Stream stream{i}_;\n' for i in range(2, extra_streams+2)
    ])
    connectors['destructor'].extend([
        f'if (stream{i}_.isValid()) throw std::logic_error("[_param:class_name::~_param:class_name] Stream {i} not released");\n'
        for i in range(2, extra_streams+2)
    ])

    connectors['stream_functions_cxx'].extend([
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

def iterate_constructor(connectors: dict, size_connectors: dict, constructor: dict, language: str) -> None:
    """Iterates the constructor section and adds the necessary connectors."""
    section_creation('constructor', constructor, connectors, size_connectors)
    connectors['host_members'] = []
    for key,item_type in constructor.items():
        device_item = f'_{key}_d'
        host_item = f'_{key}_h'
        size_item = f'SIZE_{key.upper()}'
        if key != 'nTiles':
            connectors['constructor_args'].append( f'{item_type} {key}' )
            connectors['host_members'].append( f'{host_item}' )
        connectors['public_members'].extend(
            [f'{item_type} {host_item};\n', f'{item_type}* {device_item};\n']
        )
        connectors['set_members'].extend(
            [f'{host_item}{"{tiles_.size()}" if key == "nTiles" else f"{{{key}}}"}', 
             f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = pad( sizeof({item_type}) );\n'
        )
        set_pointer_determination(connectors, 'constructor', key, device_item, size_item, item_type)
        connectors['memcpy_constructor'].append(
            f'std::memcpy({key}_p, static_cast<void*>(&{host_item}), {size_item});\n'
        )

def tmdata_memcpy_f(connectors: dict, host: str, construct: str, data_type: str, pinned: str, use_ref: str, size: str, item):
    connectors['memcpy_tilemetadata'].extend([
        f'{data_type} {host}{construct};\n',
        f'char_ptr = static_cast<char*>( static_cast<void*>({pinned}) ) + n * {size};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({use_ref}{host}), {size});\n\n',
    ])

def iterate_tilemetadata(connectors: dict, size_connectors: dict, tilemetadata: dict, language: str, num_arrays: int) -> None:
    section_creation('tilemetadata', tilemetadata, connectors, size_connectors)
    connectors['tile_descriptor'] = []
    missing_dependencies = cpp_helpers.get_metadata_dependencies(tilemetadata, language)
    use_ref = ""
    memcpy_func = tmdata_memcpy_f if language == consts.Language.fortran else cpp_helpers.tmdata_memcpy_cpp
    if language == consts.Language.cpp: 
        cpp_helpers.insert_farray_size(size_connectors, num_arrays)
        use_ref = "&"
    for item,name in tilemetadata.items():
        try:
            if language == consts.Language.fortran:
                item_type = consts.tile_variable_mapping[name]
            else: 
                item_type = consts.cpp_equiv[consts.tile_variable_mapping[name]]
        except Exception:
            warnings.warn(f"{name} was not found in tile_variable_mapping. Ignoring...")
            continue
        
        device_item = f'_{name}_d'
        pinned_item = f'{name}_p'
        size_item = f'SIZE_{item.upper()}'
        host_item = f'{name}_h'
        # Be careful! MDIM is 3 in Flash-X but might not always be 3 in every language
        size_eq = f"3 * sizeof({item_type})" if language == consts.Language.fortran else f"sizeof({ consts.farray_mapping.get(item_type, item_type) })"

        connectors['public_members'].extend(
            [f'{item_type}* {device_item};\n']
        )
        connectors['set_members'].extend(
            [f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {size_eq};\n'
        )
        set_pointer_determination(connectors, 'tilemetadata', name, device_item, f'_nTiles_h * {size_item}', item_type)
        item_type = consts.farray_mapping.get(item_type, item_type)
        connectors['tile_descriptor'].append(
            f'const {item_type} {name} = tileDesc_h->{name}();\n'
        )

        if 'unsigned ' in item_type:
            item_type = item_type.replace('unsigned ', '')
            construct_host = f' = static_cast<{item_type}>({host_item})'
            use_ref = "&"
        else:
            base_type = consts.tile_variable_mapping[name]
            fix_index = "+1" if 'int' in base_type else ""
            construct_host = f"[3] = {{ {name}.I(){fix_index}, {name}.J(){fix_index}, {name}.K(){fix_index} }}"
            item_type = consts.tile_variable_mapping[name]
            use_ref = ""

        # NO
        memcpy_func(connectors, host_item, construct_host, item_type, pinned_item, use_ref, size_item, name)
 
    connectors['tile_descriptor'].extend([
        f'const {consts.farray_mapping[consts.tile_variable_mapping[item]]} {item} = tileDesc_h->{item}();\n'
        for item in missing_dependencies
    ]) 

def iterate_lbound(connectors: dict, size_connectors: dict, lbound: dict, lang: str):
    """Iterates the lbound section. """
    dtype = 'int' if lang == consts.Language.fortran else 'IntVect'
    dtype_size = '3 * sizeof(int)' if lang == consts.Language.fortran else 'IntVect'
    memcpy_func = tmdata_memcpy_f if lang == consts.Language.fortran else cpp_helpers.tmdata_memcpy_cpp
    use_ref = '' if lang == consts.Language.fortran else '&'
    lbound_mdata = ' + '.join( f'SIZE_{item.upper()}' for item in lbound ) if lbound else '0'
    size_connectors['size_tilemetadata'] = ' + '.join( [size_connectors['size_tilemetadata'], lbound_mdata])
    for key,bound in lbound.items():
        device_item = f'_{key}_d'
        pinned = f'{key}_p'
        size_item = f'SIZE_{key.upper()}'
        constructor_expression,memcpy_list = consts.format_lbound_string(key, bound)

        connectors['public_members'].append( f'{dtype}* {device_item};\n' )
        connectors['set_members'].append( f'{device_item}{{nullptr}}')
        connectors['size_determination'].append( f'constexpr std::size_t {size_item} = sizeof({dtype_size});\n' )
        connectors['pointers_tilemetadata'].append(
            f"""{dtype}* {pinned} = static_cast<{dtype}*>( static_cast<void*>(ptr_p) );\n""" + 
            f"""{device_item} = static_cast<{dtype}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_p += _nTiles_h * {size_item};\n""" + 
            f"""ptr_d += _nTiles_h * {size_item};\n\n"""
        )
        connectors['tile_descriptor'].append(
            f'const IntVect {key} = {constructor_expression};\n'
        )
        memcpy_func(connectors, f'{key}_h', 
                    f"[{len(memcpy_list)}] = {{{','.join(memcpy_list)}}}",
                    dtype, pinned, use_ref, size_item, key)


def iterate_tilein(connectors: dict, size_connectors: dict, tilein: dict, params:dict, language: str) -> None:
    section_creation('tilein', tilein, connectors, size_connectors)
    pinnedLocation = set()
    for item,data in tilein.items():
        device_item = f'_{item}_d'
        pinned_item = f'{item}_p'
        size_item = f'SIZE_{item.upper()}'
        extents = data['extents']
        start = data['start']
        end = data['end']
        raw_type = data['type']
        item_type = data['type'] if language == consts.Language.fortran else consts.cpp_equiv[data['type']]
        extents = ' * '.join(f'({item})' for item in data['extents'])
        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
            f'{raw_type}* {pinned_item};\n'
        )
        connectors['set_members'].append(
            [f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({end} - {start} + 1) * sizeof({item_type});\n'
        )
        connectors['pinned_sizes'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({end} - {start} + 1) * sizeof({item_type});\n'
        )
        set_pointer_determination(connectors, 'tilein', item, device_item, f'_nTiles_h * {size_item}', item_type, False)
        add_memcpy_connector(connectors, 'tilein', extents, item, start, end, size_item, raw_type)
        # if language == consts.Language.cpp:
        #     cpp_helpers.insert_farray_memcpy(connectors, item, , , tilein[item]['extents'][-1], item_type)
    connectors['memcpy_tileinout'].extend(pinnedLocation)

def add_memcpy_connector(connectors: dict, section: str, extents, item, start, end, size_item, raw_type):
    connectors[f'memcpy_{section.replace("-", "")}'].extend([
        f'{raw_type}* {item}_d = tileDesc_h->dataPtr();\n'
        f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
        f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});\n'
        f'char_ptr = static_cast<char*>( static_cast<void*>({item}_p) ) + n * {size_item};\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({item}_d + offset_{item}), nBytes_{item});\n'
    ])
    connectors['in_pointers'].append(f'{raw_type}* {item}_data_h = tileDesc_h->dataPtr();\n')

def add_unpack_connector(connectors: dict, section: str, extents, item, start, end, item_type, raw_type):
    connectors[f'unpack_{section.replace("-", "")}'].extend([
        f'{raw_type}* {item}_d = tileDesc_h->dataPtr();\n'
        f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
        f'Real*        start_h_{item} = {item}_data_h + offset_{item};\n'
        f'const Real*  start_p_{item} = {item}_data_p + offset_{item};\n'
        f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type});\n',
        f'std::memcpy(static_cast<void*>(start_h_{item}), static_cast<const void*>(start_p_{item}), nBytes_{item});\n'
    ])
    connectors['out_pointers'].append(f'{raw_type}* {item}_data_p = {item}_p + n * SIZE_{item};\n')

def iterate_tileinout(connectors: dict, size_connectors: dict, tileinout: dict, params:dict, language: str) -> None:
    section_creation('tileinout', tileinout, connectors, size_connectors)
    connectors['memcpy_tileinout'] = []
    connectors['unpack_tileinout'] = []
    pinnedLocation = set()
    for item,value in tileinout.items():
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        pinned_item = f'{item}_p'
        raw_type = value['type']
        item_type = value['type'] if language == consts.Language.fortran else consts.cpp_equiv[value['type']]
        start_in = value['start-in']
        end_in = value['end-in']
        start_out = value['start-out']
        end_out = value['end-out']
        extents = ' * '.join(f'({item})' for item in value['extents'])
        unks = f'{end_in} - {start_in} + 1'
        connectors['public_members'].append(
            f'{raw_type}* {device_item};\n'
            f'{raw_type}* {pinned_item};\n'
        )
        connectors['set_members'].extend(
            [f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({unks}) * sizeof({raw_type});\n'
        )
        connectors['pinned_sizes'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({unks}) * sizeof({raw_type});\n'
        )
        set_pointer_determination(connectors, 'tileinout', item, device_item, f'_nTiles_h * {size_item}', raw_type, False)
        add_memcpy_connector(connectors, 'tileinout', extents, item, start_in, end_in, size_item, raw_type)
        add_unpack_connector(connectors, 'tileinout', extents, item, start_out, end_out, item_type, raw_type)
        if language == consts.Language.cpp:
            # hardcode lo and hi for now.
            cpp_helpers.insert_farray_memcpy(connectors, item, "loGC", "hiGC", unks, raw_type)
    connectors['memcpy_tileinout'].extend(pinnedLocation)

def iterate_tileout(connectors: dict, size_connectors: dict, tileout: dict, params:dict, language: str) -> None:
    section_creation('tileout', tileout, connectors, size_connectors)
    connectors['unpack_tileout'] = []
    for item,data in tileout.items():
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        start = data['start']
        end = data['end']
        extents = ' * '.join(f'({item})' for item in data['extents'])
        item_type = data['type']
        raw_type = data['type']

        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
            f'{raw_type}* {item}_p;\n'
        )
        connectors['set_members'].append(
            f'{device_item}{{nullptr}}'
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type});\n'
        )
        connectors['pinned_sizes'].append(
            f'constexpr std::size_t {size_item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type});\n'
        )
        set_pointer_determination(connectors, 'tileout', item, device_item, f'_nTiles_h * {size_item}', item_type, False)
        add_unpack_connector(connectors, "tileout", extents, item, start, end, item_type, raw_type)
        # if language == consts.Language.cpp:
        #     cpp_helpers.insert_farray_memcpy(connectors, item, , , tilein[item]['extents'][-1], item_type)

def iterate_tilescratch(connectors: dict, size_connectors: dict, tilescratch: dict, language: str) -> None:
    section_creation('tilescratch', tilescratch, connectors, size_connectors)
    for item in tilescratch:
        lbound = f"lo{item[0].capitalize()}{item[1:]}"
        # hbound = ...
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        extents = ' * '.join(f'({item})' for item in tilescratch[item]['extents'])
        item_type = tilescratch[item]['type']
        connectors['public_members'].append( f'{item_type}* {device_item};\n' )
        connectors['set_members'].append( f'{device_item}{{nullptr}}' )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * sizeof({item_type});\n'
        )
        connectors['pointers_tilescratch'].append(
            f"""{device_item} = static_cast<{item_type}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_d+= _nTiles_h * {size_item};\n\n"""
        )

        if language == consts.Language.cpp:
            # tempoarary until bounds sections are solidified 
            bound_map = {
                'auxC': ['loGC', 'hiGC', '1'],
                'flX': ['lo', 'IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }', '5'],
                'flY': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }', '5'],
                'flZ': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }', '1']
            }
            cpp_helpers.insert_farray_memcpy(connectors, item, bound_map[item][0], bound_map[item][1], bound_map[item][2], item_type)

def sort_dict(section, sort_key) -> dict:
    return dict( sorted(section, key = sort_key, reverse = True) )

def write_connectors(template, connectors):
    """Writes connectors to the """
    # constructor args requires a special formatting 
    template.writelines([
        f'/* _connector:constructor_args */\n'
        ] + [ ','.join(connectors['constructor_args']) ] + ['\n\n'])
    del connectors['constructor_args']

    # set members needs a special formatting
    template.writelines([
        f'/* _connector:set_members */\n'
        ] + [',\n'.join(connectors['set_members']) ] + ['\n\n'])
    del connectors['set_members']

    template.writelines(
        [ f'/* _connector:host_members */\n' ] + 
        [','.join(connectors["host_members"])] + 
        ['\n\n']
    )
    del connectors['host_members']

    # write any leftover connectors
    for connection in connectors:
        template.writelines([
            f'/* _connector:{connection} */\n'
        ] + [ ''.join(connectors[connection]) ] + ['\n'] 
    )
        
def set_default_params(data: dict, params: dict):
    """
    Sets the default parameters for cgkit.
    
    Parameters:
        data: dict - The JSON data 
        params: dict - The params dictionary
    """
    params['align_size'] = data.get("byte-align", 16)
    params['nextrastreams'] = data.get('n-extra-streams', 0)
    params['class_name'] = data["name"]
    params['i_give_up'] = data["name"]
    params['ndef_name'] = f'{data["name"].upper()}_UNIQUE_IFNDEF_H_'
    
def generate_outer(name: str, params: dict):
    """
    Generates the outer template for the datapacket template.
    
    Parameters:
        name: the name of the class.\n
        params: the parameter list to write to the outer template.
    Returns:
        None
    """
    with open(name, 'w') as outer:
        outer.writelines([
            '/* _connector:datapacket_outer */\n',
            '/* _link:datapacket */\n'
        ] + [
            '\n'.join(f'/* _param:{item} = {params[item]} */' for item in params)
        ])

def write_size_connectors(size_connectors: dict, file):
    """
    Writes the size connectors to the specified file.
    
    Parameters:
        size_connectors: the size connectors to write to file.\n
        file: the file to write to.
    Returns:
        None
    """
    for key,item in size_connectors.items():
        file.write(f'/* _connector:{key} */\n{item}\n\n')
        
#TODO: There is a large issue with size sorting. For array types, sizes are determined by sizes as well as
#      the given extents and unk vars. Since the extents are passed in as mathematical expressions, we would
#      need to write an expression parser in order to get accurate size sorting. Either that, or the expressions
#      given in the JSON file need to be single constants and not mathematical expressions
def generate_helper_template(data: dict) -> None:
    with open(data["helpers"], 'w') as template:
        size_connectors = defaultdict(str)
        connectors = defaultdict(list)
        params = defaultdict(str)

        # set defaults for the connectors. 
        connectors['constructor_args'] = []
        connectors['set_members'] = []
        connectors['size_determination'] = []
        set_default_params(data, params)

        sort_func = lambda x: sizes.get(x[1], 0) if sizes else 1
        sizes = data["sizes"]
        constructor = data.get(json_sections.GENERAL, {})
        constructor['nTiles'] = 'int' # every data packet always needs nTiles.
        constructor = constructor.items()
        iterate_constructor(connectors, size_connectors, sort_dict(constructor, sort_func), data['language'])

        num_arrays = len(data.get(json_sections.T_SCRATCH, {})) + len(data.get(json_sections.T_IN, {})) + \
                     len(data.get(json_sections.T_IN_OUT, {})) + len(data.get(json_sections.T_OUT, {}))
        sort_func = lambda x: sizes.get(consts.tile_variable_mapping[x[1]], 0) if sizes else 1
        metadata = data.get(json_sections.T_MDATA, {}).items()
        iterate_tilemetadata(connectors, size_connectors, sort_dict(metadata, sort_func), data['language'], num_arrays)

        lbound = data.get(json_sections.LBOUND, {})
        iterate_lbound(connectors, size_connectors, lbound, data['language'])

        sort_func = lambda x: sizes.get(x[1]['type'], 0) if sizes else 1
        for section,funct in {json_sections.T_IN: iterate_tilein, json_sections.T_IN_OUT: iterate_tileinout,
                              json_sections.T_OUT: iterate_tileout }.items():
            dictionary = data.get(section, {}).items()
            funct(connectors, size_connectors, sort_dict(dictionary, sort_func), params, data['language'])
        tilescratch = data.get(json_sections.T_SCRATCH, {}).items()
        iterate_tilescratch(connectors, size_connectors, sort_dict(tilescratch, sort_func), data['language'])

        if data['language'] == consts.Language.cpp: 
            cpp_helpers.insert_farray_information(data, connectors, size_connectors)

        generate_outer(data["outer"], params)
        write_size_connectors(size_connectors, template)
        generate_extra_streams_information(connectors, data.get('n-extra-streams', 0))
        write_connectors(template, connectors)

        
