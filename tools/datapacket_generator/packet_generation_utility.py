import json_sections
import re
from typing import Tuple
from enum import Enum

# source keywords
_GRID = "grid_data"
_SCRATCH = "scratch"

# TODO: Will flux arrays always contain 5 unks? Will auxC always only have 1 unk?
#       Need to determine with lbound.
# TODO TODO: These should be flash-x specific keywords. They should not be stored within
#            the data packet generation code. Anything application specific should be able to 
#            provide their own keyword list for parsing data packet jsons / yamls / whatever 
#            they want to have the data packet information stored in.
PREDEFINED_STRUCT_KEYWORDS = {
    'auxC': ['loGC', 'hiGC', '1'],
    'flX': ['lo', 'IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }', '5'],
    'flY': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }', '5'],
    'flZ': ['lo', 'IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }', '5']
}

FARRAY_MAPPING = {
    "int": "IntVect",
    "real": "RealVect"
}

F_HOST_EQUIVALENT = {
    'RealVect': 'real',
    'IntVect': 'int'
}

CPP_EQUIVALENT = {
    "real": "RealVect",
    "int": "IntVect",
    "logical": "bool"
}

TILE_VARIABLE_MAPPING = {
    'levels': 'unsigned int',
    'gridIndex': 'int',
    'tileIndex': 'int',
    'deltas': 'RealVect',
    'lo': "IntVect",
    'hi': "IntVect",
    'loGC': "IntVect",
    'hiGC': "IntVect"
}

DATA_POINTER_MAP = {
    'unk': 'tileDesc_h->dataPtr()',
    'flX': 'tileDesc_h->fluxDataPtrs()[Axis::I]',
    'flY': 'tileDesc_h->fluxDataPtrs()[Axis::J]',
    'flZ': 'tileDesc_h->fluxDataPtrs()[Axis::K]'
}

class Language(Enum):
    """Gives all possible languages for data packet generation. Included languages are fortran and cpp."""
    cpp = 'cpp'
    fortran = 'fortran'

    def __str__(self):
        return self.value
    
class _NoTaskArgumentListException(BaseException):
    """Raised when the provided JSON file does not have a task function argument list."""
    pass

class _TaskArgumentListMismatchException(BaseException):
    """Raised when the items in the task function argument list do not match the items in the JSON file."""
    pass

class _DuplicateItemException(BaseException):
    """Raised when there is a duplicate item key in the JSON file."""
    pass


def remove_invalid_parens(invalid_string: str) -> str:
    """Removes any invalid parentheses from a string."""
    stack = []
    to_remove = []

    for idx,char in enumerate(invalid_string):
        if char == "(":
            stack.append( (char, idx) )
        elif char == ")":
            try:
                stack.pop()
            except:
                to_remove.append( (char, idx) )
    to_remove.extend(stack)
    for (char,idx) in to_remove:
        invalid_string = invalid_string[:idx] + '' + invalid_string[idx + 1:]
    return invalid_string


def check_json_validity(data: dict) -> bool:
    """
    Checks if the keys in the task function argument list are all present in the JSON
    input, as well as checks if any duplicates exist between JSON keys. 

    :param dict data: The JSON to check the validity of.
    :return: True if the JSON is valid, False otherwise.
    :rtype: bool
    """
    task_arguments = data.get(json_sections.ORDER, [])
    if not task_arguments:
        raise _NoTaskArgumentListException("Missing task_function_argument_list.")
    all_items_list = [ set(data[section]) for section in json_sections.ALL_SECTIONS if section in data ]
    
    # This checks if there is a duplicate between any 2 sets, out of n total sets. 
    # Is there a faster way to do this using set operations? 
    all_items_dupe = list(all_items_list)
    duplicates = set()
    for set1 in all_items_list:
        for set2 in all_items_dupe:
            if set1 is not set2:
                duplicates = duplicates.union( set1.intersection(set2) )
        all_items_dupe.remove(set1)

    if duplicates:
        raise _DuplicateItemException(f"There is a duplicate item key in the JSON. Duplicates: {duplicates}")
    missing_items = set.union(*all_items_list) ^ set(task_arguments)
    if missing_items:
        raise _TaskArgumentListMismatchException(f"task_function_argument_list items do not match the items in the JSON. Missing: {missing_items}")
    return True