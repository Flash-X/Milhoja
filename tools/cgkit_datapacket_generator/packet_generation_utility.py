from enum import Enum
import json_sections
from typing import Tuple

GENERATED_MESSAGE = "This code was generated with the data packet generator."

F_LICENSE_BLOCK = """!> @copyright Copyright 2022 UChicago Argonne, LLC and contributors
!!
!! @licenseblock
!! Licensed under the Apache License, Version 2.0 (the "License");
!! you may not use this file except in compliance with the License.
!!
!! Unless required by applicable law or agreed to in writing, software
!! distributed under the License is distributed on an "AS IS" BASIS,
!! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!! See the License for the specific language governing permissions and
!! limitations under the License.
!! @endlicenseblock
!!
!! @file\n
"""

farray_mapping = {
    "int": "IntVect",
    "real": "RealVect"
}

cpp_equiv = {
    "real": "RealVect",
    "int": "IntVect"
}

tile_variable_mapping = {
    'levels': 'unsigned int',
    'gridIndex': 'int',
    'tileIndex': 'int',
    'deltas': 'real',
    'lo': "int",
    'hi': "int",
    'loGC': "int",
    'hiGC': "int"
}

class Language(Enum):
    """Gives all possible languages for data packet generation. Included languages are fortran and cpp."""
    cpp = 'cpp'
    fortran = 'fortran'

    def __str__(self):
        return self.value
    
class NoTaskArgumentListExcepiton(BaseException):
    """Raised when the provided JSON file does not have a task function argument list."""
    pass

class TaskArgumentListMismatchException(BaseException):
    """Raised when the items in the task function argument list do not match the items in the JSON file."""
    pass

class DuplicateItemException(BaseException):
    """Raised when there is a duplicate item key in the JSON file."""
    pass

def format_lbound_string(name:str, lbound: list) -> Tuple[str, list]:
    """
    Given an lbound string, it formats it and returns the formatted string, as well as a list of the necessary lbound construction arguments.
    
    :param str name: The lbound string.
    :param list lbound: The lbound string split up as a list.
    :return: A tuple containing the formatted lbound string, as well as a list of formatted lbound items.
    :rtype: Tuple[str, list]
    """
    lbound_list = []
    formatted = ""
    for item in lbound:
        try:
            lbound_list.append(str(int(item)))
        except Exception:
            formatted = item
            tile_name = formatted.split(' ')[0].replace('tile_', '')
            tile_name = f'{tile_name}()'
            start_location = item.find('(')
            formatted = f'tileDesc_h->{tile_name}'
            if start_location != -1:
                formatted = f'tileDesc_h->{tile_name} - IntVect{{ LIST_NDIM{item[start_location:]} }}'
            lbound_list.extend( [f'{name}.I() + 1', f'{name}.J() + 1', f'{name}.K() + 1'] )
    return (formatted, lbound_list)
    
def check_json_validity(data: dict) -> bool:
    """
    Checks if the keys in the task function argument list are all present in the JSON
    input, as well as checks if any duplicates exist between JSON keys. 

    :param dict data: The JSON to check the validity of.
    :return: True if the JSON is valid, False otherwise.
    :rtype: bool
    """
    task_arguments = data.get(json_sections.ORDER, {})
    if not task_arguments:
        raise NoTaskArgumentListExcepiton("Missing task-function-argument-list.")
    all_items_list = [ set(data[section]) if section != json_sections.T_MDATA else set(data[section].values()) for section in json_sections.ALL_SECTIONS if section in data ]
    
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
        raise DuplicateItemException(f"There is a duplicate item key in the JSON. Duplicates: {duplicates}")
    missing_items = set.union(*all_items_list) ^ set(task_arguments)
    if missing_items:
        raise TaskArgumentListMismatchException(f"task-function-argument-list items do not match the items in the JSON. Missing: {missing_items}")
    return True