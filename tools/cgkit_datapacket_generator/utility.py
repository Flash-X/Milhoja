from enum import Enum
import json_sections

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

C_LICENSE_BLOCK = """
/**
 * @copyright Copyright 2022 UChicago Argonne, LLC and contributors
 *
 * @licenseblock
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * @endlicenseblock
 *
 * @file
 */
 """
farray_mapping = {
    "int": "IntVect",
    "Real": "RealVect"
}

cpp_equiv = {
    "Real": "RealVect",
    "int": "IntVect"
}

finterface_constructor_args = {
    'cc': "loGC, hiGC",
    'fcx': "lo, IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }",
    'fcy': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }",
    'fcz': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }"
}

tile_variable_mapping = {
    'levels': 'unsigned int',
    'gridIndex': 'int',
    'tileIndex': 'int',
    'nCcVariables': 'unsigned int',
    'nFluxVariables': 'unsigned int',
    'deltas': 'Real',
    'lo': "int",
    'hi': "int",
    'loGC': "int",
    'hiGC': "int",
    'data': 'FArray4D',
    'dataPtr': 'Real*',
}

class Language(Enum):
    cpp = 'cpp'
    fortran = 'fortran'

    def __str__(self):
        return self.value
    
class NoTaskArgumentListExcepiton(BaseException):
    pass

class TaskArgumentListMismatchException(BaseException):
    pass

class DuplicateItemException(BaseException):
    pass
    
def check_json_validity(data: dict) -> bool:
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
        raise NoTaskArgumentListExcepiton("Missing task-function-argument-list.")
    all_items_list = [ set(data[section]) if section != json_sections.T_MDATA else set(data[section].values()) for section in json_sections.ALL_SECTIONS if section in data ]
    if set.intersection( *all_items_list ):
        raise DuplicateItemException("There is a duplicate item key in the JSON. Please ensure all item keys are unique.")
    all_items_list = set.union(*all_items_list)
    if all_items_list ^ set(task_arguments):
        raise TaskArgumentListMismatchException("task-function-argument-list items do not match the items in the JSON.")
    return True