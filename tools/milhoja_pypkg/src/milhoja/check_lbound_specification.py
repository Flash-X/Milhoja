from .constants import (
    GRID_DATA_ARGUMENT,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT
)
from .LogicError import LogicError


def check_lbound_specification(arg, arg_specs_all, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    :param arg: Name of argument
    :param arg_specs_all: Specifications for all arguments in the argument list
        of the given argument's subroutine
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    # LOG_NAME = ERROR_CHECK_LOG_TAG

    # ----- LBOUND SPECIFICATIONS
    arg_spec = arg_specs_all[arg]

    expected = {"source", "array"}
    actual = set(arg_spec)
    if actual != expected:
        msg = f"Invalid lbound specification keys for {arg} ({actual})"
        raise ValueError(msg)

    target_array = arg_spec["array"]
    if not isinstance(target_array, str):
        msg = f"{arg}'s array name is not string ({target_array})"
        raise TypeError(msg)
    elif target_array == "":
        raise ValueError(f"lbound {arg}'s array name empty")
    elif target_array == arg:
        raise LogicError(f"lbound {arg}'s array cannot be itself")
    elif target_array not in arg_specs_all:
        msg = "{} lbound's array {} not in subroutine argument list"
        raise ValueError(msg.format(arg, target_array))

    # Except for external, all of these must be arrays.  If another type is
    # added that need not be an array, then evaluate need to update error
    # checking in check_group_specification.
    ALLOWABLES = [
        EXTERNAL_ARGUMENT,
        SCRATCH_ARGUMENT,
        GRID_DATA_ARGUMENT,
        TILE_COORDINATES_ARGUMENT,
        TILE_FACE_AREAS_ARGUMENT,
        TILE_CELL_VOLUMES_ARGUMENT
    ]
    target_source = arg_specs_all[target_array]["source"]
    if target_source not in ALLOWABLES:
        msg = "lbound {} can't be obtained for array of type {}"
        raise ValueError(msg.format(arg, target_source))
