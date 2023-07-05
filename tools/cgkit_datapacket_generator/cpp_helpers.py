"""A collection of alternate functions to be used when generating cpp packets."""
import utility as util
import json_sections

def insert_farray_size_constructor(size_connector: dict):
    packet_contents_size = "_nTiles_h * sizeof(PacketContents)"
    size_connector['size_constructor'] = f'{size_connector["size_constructor"]} + {packet_contents_size}'

def get_metadata_dependencies(metadata: dict, language: str) -> set:
    if language == util.Language.fortran: return set()
    mdata_set = set(metadata.values())
    return mdata_set.symmetric_difference( {"lo", "hi", "loGC", "hiGC"} ).intersection({"lo", "hi", "loGC", "hiGC"})

def insert_farray_size(connectors: dict, num_arrays: int) -> None:
    line = connectors['size_tilemetadata']
    insert_index = line.find('(')
    connectors['size_tilemetadata'] = f'{line[:insert_index + 1]}({num_arrays} * sizeof(FArray4D)) + {line[insert_index + 1:]}'

def insert_farray_memcpy(connectors: dict, item: str, unks: int, data_type: str):
    connectors['pointers_tilemetadata'].append(
        f"""char* {item}_fa4_p = ptr_p;\n""" + \
        f"""char* {item}_fa4_d = ptr_d;\n""" + \
        f"""ptr_p += _nTiles_h * sizeof(FArray4D);\n""" + \
        f"""ptr_d += _nTiles_h * sizeof(FArray4D);\n\n"""
    )
    # does not matter memcpy to insert into
    connectors['memcpy_tilein'].extend([
        f'char_ptr = {item}_fa4_d + n * sizeof(FArray4D);\n',
        f'tilePtrs_p->{item}_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );\n'
        f"""FArray4D {item}_d{{ static_cast<{data_type}*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_{item}_d) ) """ + \
        f"""+ n * SIZE_{item.upper()})) }};\n""",
        f'char_ptr = {item}_fa4_p + n * sizeof(FArray4D);\n',
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&{item}_d), sizeof(FArray4D));\n\n'
    ])

def insert_farray_information(data: dict, connectors: dict, size_connectors) -> None:
    insert_farray_size_constructor(size_connectors)
    farray_pointers = list(data.get(json_sections.T_IN, {}).items()) + \
                      list(data.get(json_sections.T_IN_OUT, {}).values()) + \
                      list(data.get(json_sections.T_OUT, {}).values()) + \
                      list(data.get(json_sections.T_SCRATCH, {}).values())
    print(farray_pointers)

def tmdata_memcpy_cpp(connectors: dict, host, construct, data_type, pinned, use_ref, size, item):
    dtype = util.cpp_equiv[data_type] if data_type in util.cpp_equiv else data_type
    connectors['memcpy_tilemetadata'].extend([
        f"""char_ptr = static_cast<char*>( static_cast<void*>( {pinned} ) ) + n * {size};\n""",
        f"""tilePtrs_p->{item}_d = static_cast<{dtype}*>( static_cast<void*>(char_ptr) );\n""",
        f"""std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&{item}), {size});\n\n"""
    ])