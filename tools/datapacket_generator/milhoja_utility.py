import sys
import warnings
from enum import Enum
from typing import Tuple
from typing import Union
import json_sections

class Language(Enum):
    cpp = 'cpp'
    fortran = 'fortran'

    def __str__(self):
        return self.value

tile_known_types = {
    'levels': 'unsigned int',
    'gridIndex': 'int',
    'tileIndex': 'int',
    'nCcVariables': 'unsigned int',
    'nFluxVariables': 'unsigned int',
    'deltas': 'RealVect',
    'lo': "IntVect",
    'hi': "IntVect",
    'loGC': "IntVect",
    'hiGC': "IntVect",
    'data': 'FArray4D',
    'dataPtr': 'Real*',
}

cpp_equiv = {
    "RealVect": "Real",
    "IntVect": "int"
}

imap = {
    'IntVect':  '<Milhoja_IntVect.h>',
    'Real':     '<Milhoja_real.h>',
    'RealVect': '<Milhoja_RealVect.h>',
    'FArray1D': '<Milhoja_FArray1D.h>',
    'FArray2D': '<Milhoja_FArray2D.h>',
    'FArray3D': '<Milhoja_FArray3D.h>',
    'FArray4D': '<Milhoja_FArray4D.h>'
}

# NOTE: These are the fortran spaces, while cpp size map is specifically for use with cpp.
fortran_size_map = {
    'cc': "nxbGC_h * nybGC_h * nzbGC_h * ({unk}) * sizeof({size})",
    'fcx': "(nxbGC_h+1) * nybGC_h * nzbGC_h * ({unk}) * sizeof({size})",
    'fcy': "nxbGC_h * (nybGC_h+1) * nzbGC_h * ({unk}) * sizeof({size})",
    'fcz': "nxbGC_h * nybGC_h * (nzbGC_h+1) * ({unk}) * sizeof({size})"
}

cpp_size_map = {
    'cc': "(nxb_+2*{guard}*MILHOJA_K1D) * (nyb_+2*{guard}*MILHOJA_K2D) * (nzb_+2*{guard}*MILHOJA_K3D) * ({unk}) * sizeof({size})",
    'fcx': "((nxb_+1)+2*{guard}*MILHOJA_K1D) * ((nyb_)+2*{guard}*MILHOJA_K2D) * ((nzb_)+2*{guard}*MILHOJA_K3D) * ({unk}) * sizeof({size})",
    'fcy': "((nxb_)+2*{guard}*MILHOJA_K1D) * ((nyb_+1)+2*{guard}*MILHOJA_K2D) * ((nzb_)+2*{guard}*MILHOJA_K3D) * ({unk}) * sizeof({size})",
    'fcz': "((nxb_)+2*{guard}*MILHOJA_K1D) * ((nyb_)+2*{guard}*MILHOJA_K2D) * ((nzb_+1)+2*{guard}*MILHOJA_K3D) * ({unk}) * sizeof({size})"
}

finterface_constructor_args = {
    'cc': "loGC, hiGC",
    'fcx': "lo, IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }",
    'fcy': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }",
    'fcz': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }"
}

# Government sanctioned constants.
constants = {
    "NGUARD": "nGuard_",
    "NFLUXES": "nFluxVars_",
    "NUNKVAR": "nCcVars_"
}

def get_nguard(extents: list):
    """Gets the number of guard cells from the extents array."""
    nguard = extents[1].strip()
    try:
        nguard = int(nguard)
    except Exception:
        nguard = constants.get(nguard.upper(), -1)
        if nguard == -1:
            print("Constant not found")
            exit(-1)
    return nguard


def parse_extents(extents, start, end, size, language=Language.cpp) -> Tuple[str, str, str, list]:
    """
    Parses the extents string found in the packet JSON file.

    Parameters:
        extents - The string to parse\n
        start - the starting index of the array\n
        end - the ending index of the array\n
        size - the variable type
    Returns:
        str - extents string\n
        str - nunkvars string\n
        str - indexer type\n
        list - The number of elements per var
    """
    # check if extents is a string or or an enumerable
    if isinstance(extents, str):
        if extents[-1] != ')':
            # TODO: This is not sufficient for checking adequate parenthesis
            print(f'{extents} is not closed properly.')
            exit(-1)

        extents = extents[:-1]
        sp = extents.split('(')
        indexer = sp[0]
        nguard = get_nguard(sp)

        if language == Language.cpp:
            parsed_exts = cpp_size_map[indexer].format(guard=nguard, unk=f"( ({end}) - ({start}) + 1 )", size=size)
        elif language == Language.fortran:
            parsed_exts = fortran_size_map[indexer].format(unk=f"( ({end}) - ({start}) + 1 )", size=size)

        parsed_exts = cpp_size_map[indexer].format(guard=nguard, unk=f"( ({end}) - ({start}) + 1 )", size=size)
        num_elems_per_arr = parsed_exts.split(' * ')[:-2]#[ item.replace('(', '').replace(')', '') for item in parsed_exts.split(' * ')[:-2] ]
        return parsed_exts, f"( ({end}) - ({start}) + 1 )", indexer, num_elems_per_arr

    elif isinstance(extents, list):
        parsed_extents = "(" + ' * '.join([str(item) for item in extents]) + f'){ "" if size == "" else f" * sizeof({size})" }'
        return parsed_extents, extents[-1], None, extents[:-1]
    print("Extents is not a string or list of numbers. Please refer to the documentation.")
    exit(-1)

def check_task_argument_list(data: dict) -> bool:
    """
    Checks if the keys in the task function argument list are all present in the JSON
    input, as well as checks if any duplicates exist between JSON keys. 

    Parameters:
        data The JSON dictionary
    Returns:
        - True if both all items in the JSON are present in the list and there are no duplicates.
    """
    task_arguments = data.get(json_sections.ORDER, {})
    if not task_arguments:
        print("Missing task argument list!")
        return False
    all_items_list = [ set(data[section]) for section in json_sections.ALL_SECTIONS if section in data ]
    if set.intersection( *all_items_list ):
        print("Duplicate items in JSON!")
        return False
    all_items_list = set.union(*all_items_list)
    if all_items_list ^ set(task_arguments):
        print("Missing arguments in sections or argument list.")
    return True
