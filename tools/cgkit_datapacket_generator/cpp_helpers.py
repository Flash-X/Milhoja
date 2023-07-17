"""A collection of alternate functions to be used when generating cpp packets."""
import utility as util
import json_sections

# TODO: Once bounds are properly introduced into the JSON this is no longer needed.
def get_metadata_dependencies(metadata: dict, language: str) -> set:
    """Insert metadata dependencies into the memcpy section for a cpp packet."""
    if language == util.Language.fortran: return set()
    mdata_set = set(metadata.values())
    return mdata_set.symmetric_difference( {"lo", "hi", "loGC", "hiGC"} ).intersection({"lo", "hi", "loGC", "hiGC"})

def insert_farray_size(connectors: dict, num_arrays: int) -> None:
    """Inserts the total size needed to store all farray pointers."""
    line = connectors['size_tilemetadata']
    insert_index = line.find('(')
    connectors['size_tilemetadata'] = f'{line[:insert_index + 1]}({num_arrays} * sizeof(FArray4D)) + {line[insert_index + 1:]}'

def insert_farray_memcpy(connectors: dict, item: str, lo:str, hi:str, unks: str, data_type: str):
    """
    Insers the farray memcpy and data pointer sections into the data packet connectors.

    Parameters:
        connectors: dict - Dictionary that stores connectors.
        item: str - The name of the item to be stored
        lo: str - The low bound of the array
        hi: str - the high bound of the array
        unks: str - the number of unknowns
        data_type: str - The data type of the item.
    """
    connectors['pointers_tilemetadata'].append(
        f"""_f4_{item}_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );\n""" + \
        f"""_f4_{item}_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );\n""" + \
        f"""ptr_p += _nTiles_h * sizeof(FArray4D);\n""" + \
        f"""ptr_d += _nTiles_h * sizeof(FArray4D);\n\n"""
    )
    # does not matter memcpy to insert into
    connectors['memcpy_tilein'].extend([
        # f'char_ptr = static_cast<char*>( static_cast<void*>(_f4_{item}_d) ) + n * sizeof(FArray4D);\n',
        f"""FArray4D {item}_device{{ static_cast<{data_type}*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_{item}_d) ) """ + \
        f"""+ n * SIZE_{item.upper()})), {lo}, {hi}, {unks}}};\n""",
        f'char_ptr = static_cast<char*>( static_cast<void*>(_f4_{item}_p) ) + n * sizeof(FArray4D);\n'
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&{item}_device), sizeof(FArray4D));\n\n'
    ])

# can probably shrink this function and insert it into each data section.
def insert_farray_information(data: dict, connectors: dict, size_connectors) -> None:
    """Inserts farray items into the data packet."""
    # insert_farray_size_constructor(size_connectors)
    dicts = [data.get(json_sections.T_IN, {}), data.get(json_sections.T_IN_OUT, {}), data.get(json_sections.T_OUT, {}), data.get(json_sections.T_SCRATCH, {})]
    farrays = {item: sect[item] for sect in dicts for item in sect}
    connectors['public_members'].extend([
        f'FArray4D* _f4_{item}_d;\n' for item in farrays
    ] + [
        f'FArray4D* _f4_{item}_p;\n' for item in farrays
    ])

def tmdata_memcpy_cpp(connectors: dict, host, construct, data_type, pinned, use_ref, size, item):
    """Inserts the memcpy portion for tile metadata. Various arguments are unused to share a function call with another func."""
    connectors['memcpy_tilemetadata'].extend([
        f"""char_ptr = static_cast<char*>( static_cast<void*>( {pinned} ) ) + n * {size};\n""",
        f"""std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&{item}), {size});\n\n"""
    ])