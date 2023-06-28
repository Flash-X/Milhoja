from collections import defaultdict
import utility as consts
import warnings
import cpp_alternatives

def add_size_parameter(name: str, section_dict: dict, connectors: dict):
    """
    Adds a size _param to the params dict that is passed in.

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

def set_pointer_determination(connectors: dict, section: str, item:str, device_item: str, size_item: str, item_type: str):
    connectors[f'pointers_{section}'].append(
        f"""{item_type}* {item}_p = static_cast<{item_type}*>( static_cast<void*>(ptr_p) );\n""" + 
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

def iterate_constructor(connectors: dict, size_connectors: dict, constructor: dict) -> None:
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

def get_metadata_dependencies(metadata: dict, language: str) -> set:
    if language == consts.Language.fortran: return set()
    mdata_set = set(metadata.values())
    return mdata_set.symmetric_difference( {"lo", "hi", "loGC", "hiGC"} ).intersection({"lo", "hi", "loGC", "hiGC"})

def insert_farray_information(connectors: dict, language: str) -> None:
    if language == consts.Language.fortran: return
    line = connectors['size_tilemetadata']
    insert_index = line.find('(')
    connectors['size_tilemetadata'] = f'{line[:insert_index + 1]}(5 * sizeof(FArray4D)) + {line[insert_index + 1:]}'

def insert_farray_pointers(connectors: dict, language: str) -> None:
    if language == consts.Language.fortran: return

def iterate_tilemetadata(connectors: dict, size_connectors: dict, tilemetadata: dict, params:dict, language: str) -> None:
    section_creation('tilemetadata', tilemetadata, connectors, size_connectors)
    connectors['tile_descriptor'] = []
    missing_dependencies = get_metadata_dependencies(tilemetadata, language)
    insert_farray_information(size_connectors, language)
    for item in tilemetadata:
        try:
            item_type = consts.tile_variable_mapping[tilemetadata[item]]
        except Exception:
            warnings.warn(f"{tilemetadata[item]} was not found in tile_variable_mapping. Ignoring...")
            continue
        
        device_item = f'_{tilemetadata[item]}_d'
        pinned_item = f'{tilemetadata[item]}_p'
        size_item = f'SIZE_{item.upper()}'
        host_item = f'{tilemetadata[item]}_h'
        size_eq = f"3 * sizeof({item_type})" if language == consts.Language.fortran else f"sizeof({ consts.farray_mapping.get(item_type, item_type) })"

        connectors['public_members'].extend(
            [f'{item_type}* {device_item};\n']
            # [f'std::size_t {device_item};\n']
        )
        connectors['set_members'].extend(
            [f'{device_item}{{nullptr}}']
            # [f'{device_item}{{0}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {size_eq};\n'
        )
        set_pointer_determination(connectors, 'tilemetadata', tilemetadata[item], device_item, f'_nTiles_h * {size_item}', item_type)
        item_type = consts.farray_mapping.get(item_type, item_type)
        connectors['tile_descriptor'].append(
            f'const {item_type} {tilemetadata[item]} = tileDesc_h->{tilemetadata[item]}();\n'
        )

        use_ref = ""
        if 'unsigned ' in item_type:
            item_type = item_type.replace('unsigned ', '')
            construct_host = f' = static_cast<{item_type}>({host_item})'
            use_ref = "&"
        else:
            base_type = consts.tile_variable_mapping[tilemetadata[item]]
            fix_index = "+1" if 'int' in base_type else ""
            construct_host = f"[3] = {{ {tilemetadata[item]}.I(){fix_index}, {tilemetadata[item]}.J(){fix_index}, {tilemetadata[item]}.K(){fix_index} }}"
            item_type = consts.tile_variable_mapping[tilemetadata[item]]
        if language == consts.Language.cpp:
            use_ref = "&"

        connectors['memcpy_tilemetadata'].extend([
            f'{item_type} {host_item}{construct_host};\n',
            f'char_ptr = static_cast<char*>( static_cast<void*>({pinned_item}) ) + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({use_ref}{host_item}), {size_item});\n\n',
        ])
    connectors['tile_descriptor'].extend([
        f'const {consts.farray_mapping[consts.tile_variable_mapping[item]]} {item} = tileDesc_h->{item}();\n'
        for item in missing_dependencies
    ])        

def iterate_tilein(connectors: dict, size_connectors: dict, tilein: dict, params:dict) -> None:
    section_creation('tilein', tilein, connectors, size_connectors)
    pinnedLocation = set()
    for item in tilein:
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        extents = tilein[item]['extents']
        start = tilein[item]['start']
        end = tilein[item]['end']
        item_type = tilein[item]['type']
        extents = ' * '.join(f'({item})' for item in tilein[item]['extents'])
        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
        )
        connectors['set_members'].append(
            [f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({end} - {start} + 1) * sizeof({item_type});\n'
        )
        set_pointer_determination(connectors, 'tilein', item, device_item, f'_nTiles_h * {size_item}', item_type)
        connectors['memcpy_tilein'].extend([
            f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
            f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type})'
            f'char_ptr = static_cast<char*>( static_cast<void*>({item}_p) ) + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(data_h + offset_{item}), nBytes_{item})'
        ])
        params['location_in'] = "CC1"
        pinnedLocation.add(f'pinnedPtrs_[n].{params["location_in"]}_data = static_cast<{item_type}*>( static_cast<void*>(char_ptr) );\n')
    connectors['memcpy_tileinout'].extend(pinnedLocation)

def add_unpack_connector(connectors: dict, section: str, extents, item, start, end, item_type):
    connectors[f'unpack_{section.replace("-", "")}'].extend([
        f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
        f'Real*        start_h_{item} = data_h + offset_{item};\n'
        f'const Real*  start_p_{item} = data_p + offset_{item};\n'
        f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type});\n',
        f'std::memcpy(static_cast<void*>(start_h_{item}), static_cast<const void*>(start_p_{item}), nBytes_{item});\n'
    ])

def iterate_tileinout(connectors: dict, size_connectors: dict, tileinout: dict, params:dict) -> None:
    section_creation('tileinout', tileinout, connectors, size_connectors)
    connectors['memcpy_tileinout'] = []
    connectors['unpack_tileinout'] = []
    pinnedLocation = set()
    for item in tileinout:
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        pinned_item = f'{item}_p'
        item_type = tileinout[item]['type']
        start_in = tileinout[item]['start-in']
        end_in = tileinout[item]['end-in']
        start_out = tileinout[item]['start-out']
        end_out = tileinout[item]['end-out']
        extents = ' * '.join(f'({item})' for item in tileinout[item]['extents'])
        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
        )
        connectors['set_members'].extend(
            [f'{device_item}{{nullptr}}']
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ({end_in} - {start_in} + 1) * sizeof({item_type});\n'
        )
        set_pointer_determination(connectors, 'tileinout', item, device_item, f'_nTiles_h * {size_item}', item_type)
        connectors['memcpy_tileinout'].extend([
            f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start_in});\n',
            f'std::size_t nBytes_{item} = {extents} * ( {end_in} - {start_in} + 1 ) * sizeof({item_type});\n',
            f'char_ptr = static_cast<char*>( static_cast<void*>({pinned_item}) ) + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(data_h + offset_{item}), nBytes_{item});\n'
        ])
        add_unpack_connector(connectors, 'tileinout', extents, item, start_out, end_out, item_type)

        # THis should eventually disappear.
        params['location_in'] = "CC1"
        params['location_out'] = "CC1_data"
        pinnedLocation.add(f'pinnedPtrs_[n].{params["location_in"]}_data = static_cast<{item_type}*>( static_cast<void*>(char_ptr) );\n')
    connectors['memcpy_tileinout'].extend(pinnedLocation)

def iterate_tileout(connectors: dict, size_connectors: dict, tileout: dict, params:dict) -> None:
    section_creation('tileout', tileout, connectors, size_connectors)
    connectors['unpack_tileout'] = []
    for item in tileout:
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        start = tileout[item]['start']
        end = tileout[item]['end']
        extents = ' * '.join(f'({item})' for item in tileout[item]['extents'])
        item_type = tileout[item]['type']

        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
        )
        connectors['set_members'].append(
            f'{device_item}{{nullptr}}'
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type});\n'
        )
        set_pointer_determination(connectors, 'tileout', item, device_item, f'_nTiles_h * {size_item}', item_type)
        connectors['memcpy_tileout'].extend([
            f'std::size_t offset_{item} = {extents} * static_cast<std::size_t>({start});\n',
            f'std::size_t nBytes_{item} = {extents} * ( {end} - {start} + 1 ) * sizeof({item_type})'
            f'char_ptr = static_cast<char*>( static_cast<void*>({item}_p) ) + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(data_h + offset_{item}), nBytes_{item})'
        ])
        add_unpack_connector(connectors, "tileout", extents, item, start, end, item_type)
        params['location_out'] = "CC2"

def iterate_tilescratch(connectors: dict, size_connectors: dict, tilescratch: dict) -> None:
    section_creation('tilescratch', tilescratch, connectors, size_connectors)
    for item in tilescratch:
        device_item = f'_{item}_d'
        size_item = f'SIZE_{item.upper()}'
        extents = ' * '.join(f'({item})' for item in tilescratch[item]['extents'])
        item_type = tilescratch[item]['type']
        connectors['public_members'].append(
            f'{item_type}* {device_item};\n'
        )
        connectors['set_members'].append(
            f'{device_item}{{nullptr}}'
        )
        connectors['size_determination'].append(
            f'constexpr std::size_t {size_item} = {extents} * sizeof({item_type});\n'
        )
        connectors['pointers_tilescratch'].append(
            f"""{device_item} = static_cast<{item_type}*>( static_cast<void*>(ptr_d) );\n""" + 
            f"""ptr_d+= _nTiles_h * {size_item};\n\n"""
        )

def farray_pointer_determination(connectors: dict, pointers: list):
    ...

def sort_dict(section, sort_key) -> dict:
    return dict( sorted(section, key = sort_key, reverse=True) )

def write_connectors(template, connectors):
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
        
def set_default_params(data: dict, params: dict) -> None:
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
        
        # if data['language'] == consts.Language.cpp:
        #     num_arrays = len(data.get('tile-scratch', {})) + len(data.get('tile-in', {})) + len(data.get('tile-in-out', {})) + len(data.get('tile-out', {}))
        #     params['cpp_farrays'] = f"({num_arrays} * sizeof(Farray4D)) + "
        # else:
        #     params['cpp_farrays'] = ""

        sort_func = lambda x: sizes.get(x[1], 0) if sizes else 1
        sizes = data["sizes"]
        constructor = data.get('constructor', {})
        constructor['nTiles'] = 'int' # every data packet always needs nTiles.
        constructor = constructor.items()
        iterate_constructor(connectors, size_connectors, sort_dict( constructor, sort_func) )

        sort_func = lambda x: sizes.get(consts.tile_variable_mapping[x[1]], 0) if sizes else 1
        metadata = data.get('tile-metadata', {}).items()
        iterate_tilemetadata(connectors, size_connectors, sort_dict(metadata, sort_func), params, data['language'])

        sort_func = lambda x: sizes.get(x[1]['type'], 0) if sizes else 1
        tilein = data.get('tile-in', {}).items()
        iterate_tilein(connectors, size_connectors, sort_dict(tilein, sort_func), params)
        tileinout = data.get('tile-in-out', {}).items()
        iterate_tileinout(connectors, size_connectors, sort_dict(tileinout, sort_func), params)
        tileout = data.get('tile-out', {}).items()
        iterate_tileout(connectors, size_connectors, sort_dict(tileout, sort_func), params)
        tilescratch = data.get('tile-scratch', {}).items()
        iterate_tilescratch(connectors, size_connectors, sort_dict(tilescratch, sort_func))

        generate_outer(data["outer"], params)
        write_size_connectors(size_connectors, template)
        generate_extra_streams_information(connectors, data.get('n-extra-streams', 0))
        write_connectors(template, connectors)

        
