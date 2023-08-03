"""
A collection of alternate functions to be used when generating cpp packets.

TODO: Need some way to determine when to use FArray1D/2D/3D over FArray4D.
"""
import packet_generation_utility as util
import json_sections as jsc
import DataPacketMemberVars as dpinfo

# This is a temporary measure until the bounds section in the JSON is solidified.
BOUND_MAP = {
    'auxC': ['loGC', 'hiGC', '1'],
    'flX': ['lo', 'IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }', '5'],
    'flY': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }', '5'],
    'flZ': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }', '1']
}

# TODO: Once bounds are properly introduced into the JSON this is function no longer needed.
def get_metadata_dependencies(metadata: dict, language: str) -> set:
    """
    Insert metadata dependencies into the memcpy section for a cpp packet.
    
    :param dict metadata: The dict containing tile-metadata information.
    :param str language: The language to use.
    """
    if language == util.Language.fortran: return set()
    mdata_set = set(metadata.values())
    return mdata_set.symmetric_difference( {"lo", "hi", "loGC", "hiGC"} ).intersection({"lo", "hi", "loGC", "hiGC"})

def insert_farray_size(connectors: dict, num_arrays: int) -> None:
    """Inserts the total size needed to store all farray pointers."""
    line = connectors[f'size_{jsc.T_MDATA}']
    insert_index = line.find('(')
    connectors[f'size_{jsc.T_MDATA}'] = f'{line[:insert_index + 1]}({num_arrays} * sizeof(FArray4D)) + {line[insert_index + 1:]}'

def insert_farray_memcpy(connectors: dict, item: str, lo:str, hi:str, unks: str, data_type: str):
    """
    Insers the farray memcpy and data pointer sections into the data packet connectors.

    :param dict connectors: The dict that stores all cgkit connectors.
    :param str item: The item to be stored.
    :param str lo: The low bound of the array
    :param str hi: The high bound of the array
    :param str unks: The number of unknown variables in *item*
    :param str data_type: The data type of *item*
    """
    # create metadata pointers for the f4array classes. TODO: THis is probably mentioned elsewhere
    # but when do we use FArray3D / 2D / ND? 
    connectors[f'pointers_{jsc.T_MDATA}'].append(
        f"""_f4_{item}_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );\n""" + \
        f"""_f4_{item}_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );\n""" + \
        f"""ptr_p += _nTiles_h * sizeof(FArray4D);\n""" + \
        f"""ptr_d += _nTiles_h * sizeof(FArray4D);\n\n"""
    )
    # Does not matter what memcpy section we insert into, so we default to T_IN.
    connectors[f'memcpy_{jsc.T_IN}'].extend([
        f"""FArray4D {item}_device{{ static_cast<{data_type}*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_{item}_d) ) """ + \
        f"""+ n * SIZE_{item.upper()})), {lo}, {hi}, {unks}}};\n""",
        f'char_ptr = static_cast<char*>( static_cast<void*>(_f4_{item}_p) ) + n * sizeof(FArray4D);\n'
        f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&{item}_device), sizeof(FArray4D));\n\n'
    ])

# can probably shrink this function and insert it into each data section.
def insert_farray_information(data: dict, connectors: dict, section: str) -> None:
    """
    Inserts farray items into the data packet.
    
    :param dict data: The dict containing information from the data packet JSON.
    :param dict connectors: The dict containing all cgkit connectors.
    :param str section: The connectors section to extend.
    """
    # Get all items in each data array.
    dicts = [data.get(jsc.T_IN, {}), data.get(jsc.T_IN_OUT, {}), data.get(jsc.T_OUT, {}), data.get(jsc.T_SCRATCH, {})]
    # we need to make an farray object for every possible data array
    farrays = {item: sect[item] for sect in dicts for item in sect}
    connectors[section].extend(
        [ f'FArray4D* _f4_{item}_d;\n' for item in farrays] + 
        [ f'FArray4D* _f4_{item}_p;\n' for item in farrays]
    )

# NOTE: This function call gets swapped with another so many params may be unused.
def tmdata_memcpy_cpp(connectors: dict, construct: str, use_ref: str, info: dpinfo.DataPacketMemberVars, alt_name: str):
    """
    The cpp version for the metadata memcpy section. This function contains many unused variables
    because it shares a call with the fortran version of this function.

    :param dict connectors: The dictionary containing all connectors for CGKit.
    :param str pinned: The string of the pinned item for *item*
    :param str size: The string of the size variable for *item*
    :param str item: The item to copy to pinned memory
    :param str alt_name: The name of the source pointer to be copied in.
    """
    del construct, use_ref # delete unused parameters
    """Inserts the memcpy portion for tile metadata. Various arguments are unused to share a function call with another func."""
    connectors[f'memcpy_{jsc.T_MDATA}'].extend([
        f"""char_ptr = static_cast<char*>( static_cast<void*>( {info.get_pinned()} ) ) + n * {info.get_size(False)};\n""",
        f"""std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&{alt_name}), {info.get_size(False)});\n\n"""
    ])
