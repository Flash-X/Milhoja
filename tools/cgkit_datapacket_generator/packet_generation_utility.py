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
    "auxC": ["loGC", "hiGC", "1"],
    "flX": ["lo", "IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }", "5"],
    "flY": ["lo", "IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }", "5"],
    "flZ": ["lo", "IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }", "5"],
}

FARRAY_MAPPING = {"int": "IntVect", "real": "RealVect"}

F_HOST_EQUIVALENT = {"RealVect": "real", "IntVect": "int"}

CPP_EQUIVALENT = {"real": "RealVect", "int": "IntVect", "logical": "bool"}

TILE_VARIABLE_MAPPING = {
    "levels": "unsigned int",
    "gridIndex": "int",
    "tileIndex": "int",
    "deltas": "RealVect",
    "lo": "IntVect",
    "hi": "IntVect",
    "loGC": "IntVect",
    "hiGC": "IntVect",
}

DATA_POINTER_MAP = {
    "unk": "tileDesc_h->dataPtr()",
    "flX": "tileDesc_h->fluxDataPtrs()[Axis::I]",
    "flY": "tileDesc_h->fluxDataPtrs()[Axis::J]",
    "flZ": "tileDesc_h->fluxDataPtrs()[Axis::K]",
}


class Language(Enum):
    """Gives all possible languages for data packet generation. Included languages are fortran and cpp."""

    cpp = "cpp"
    fortran = "fortran"

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

    for idx, char in enumerate(invalid_string):
        if char == "(":
            stack.append((char, idx))
        elif char == ")":
            try:
                stack.pop()
            except:
                to_remove.append((char, idx))
    to_remove.extend(stack)
    for char, idx in to_remove:
        invalid_string = invalid_string[:idx] + "" + invalid_string[idx + 1 :]
    print(invalid_string)
    return invalid_string


def parse_lbound(lbound: str, data_source: str):
    """
    Parses an lbound string for use within the generator.

    :param str lbound: The lbound string to parse.
    :param str data_source: The source of the data. Eg: scratch or grid data.
    """
    starting_index = "1"
    print(lbound)
    # data source is either grid or scratch for tile arrays.
    if lbound and data_source == _GRID:
        # We have control over the extents of grid data structures.
        lbound_info = lbound.split(",")
        # We control lbound format for grid data structures,
        # so the length of this lbound should always be 2.
        assert len(lbound_info) == 2

        # get low index
        low = lbound_info[0]
        low = low.strip().replace(")", "").replace("(", "")
        starting_index = lbound_info[-1]
        starting_index = starting_index.strip().replace("(", "").replace(")", "")
        return [low, starting_index]

    elif lbound and data_source == _SCRATCH:
        # Since tile_*** can be anywhere in scratch data I used SO solution 
        # for using negative lookahead to find tile data.
        lookahead = r",\s*(?![^()]*\))"
        matches = re.split(lookahead, lbound)
        # Can't assume lbound split is a specific size 
        # since we don't have control over structures of scratch data.
        for idx, item in enumerate(matches):
            # use this to match any int vects with only numbers
            match_intvects = r"\((?:[0-9]+[, ]*)*\)"
            unlabeled_intvects = re.findall(match_intvects, item)
            for vect in unlabeled_intvects:
                matches[idx] = item.replace(vect, f"IntVect{vect}")
        print(matches)
        return matches
    # data source was not valid.
    return [""]


def format_lbound_string(name: str, lbound: list) -> Tuple[str, list]:
    """
    TODO: Remove this.
    Given an lbound string, it formats it and returns the formatted string, 
    as well as a list of the necessary lbound construction arguments.

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
            tile_name = formatted.split(" ")[0].replace("tile_", "")
            tile_name = f"{tile_name}()"
            start_location = item.find("(")
            formatted = f"tileDesc_h->{tile_name}"
            if start_location != -1:
                formatted = f"tileDesc_h->{tile_name} - IntVect{{ LIST_NDIM{item[start_location:]} }}"
            lbound_list.extend(
                [f"{name}.I() + 1", f"{name}.J() + 1", f"{name}.K() + 1"]
            )
    return (formatted, lbound_list)


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
    all_items_list = [
        set(data[section]) for section in json_sections.ALL_SECTIONS if section in data
    ]

    # This checks if there is a duplicate between any 2 sets, out of n total sets.
    # Is there a faster way to do this using set operations?
    all_items_dupe = list(all_items_list)
    duplicates = set()
    for set1 in all_items_list:
        for set2 in all_items_dupe:
            if set1 is not set2:
                duplicates = duplicates.union(set1.intersection(set2))
        all_items_dupe.remove(set1)

    if duplicates:
        raise _DuplicateItemException(
            f"There is a duplicate item key in the JSON. Duplicates: {duplicates}"
        )
    missing_items = set.union(*all_items_list) ^ set(task_arguments)
    if missing_items:
        raise _TaskArgumentListMismatchException(
            f"task_function_argument_list items do not match the items in the JSON. Missing: {missing_items}"
        )
    return True
