from .constants import (
    LOG_LEVEL_BASIC_DEBUG,
    ERROR_CHECK_LOG_TAG,
    TILE_ARGUMENTS_ALL,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    GRID_DATA_ARGUMENT,
    THREAD_INDEX_ARGUMENT
)
from .LogicError import LogicError
from .AbcLogger import AbcLogger
from .check_tile_specification import check_tile_specification
from .check_external_specification import check_external_specification
from .check_scratch_specification import check_scratch_specification
from .check_grid_data_specification import check_grid_data_specification
from .check_thread_index_specification import check_thread_index_specification


def check_subroutine_specification(name, spec, variable_index_base, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    :param name: Name of subroutine
    :param spec: Subroutine specification obtained directly from operation
        specification
    :param variable_index_base: Minimum value in variable index set
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    LOG_NAME = ERROR_CHECK_LOG_TAG

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    expected = {
        "interface_file", "argument_list", "argument_specifications"
    }
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid set of {name} specification keys ({actual})"
        raise ValueError(msg)
    interface = spec["interface_file"]
    arg_list = spec["argument_list"]
    arg_specs_all = spec["argument_specifications"]

    if not isinstance(interface, str):
        raise TypeError(f"interface_file in {name} is not string ({interface})")
    elif interface == "":
        raise ValueError(f"Empty {name} interface filename")

    if not isinstance(arg_list, list):
        msg = f"{name}'s argument_list ({arg_list}) not a list"
        raise TypeError(msg)
    for arg in arg_list:
        if not isinstance(arg, str):
            msg = f"{name}'s argument ({arg}) not a string"
            raise TypeError(msg)
    if len(arg_list) != len(set(arg_list)):
        msg = f"Repeated names in {name}'s argument list"
        raise ValueError(msg)

    if not isinstance(arg_specs_all, dict):
        msg = f"argument_specifications ({arg_specs_all}) in {name} not dict"
        raise TypeError(msg)
    elif set(arg_list) != set(arg_specs_all):
        msg = f"Incomptabile argument list & specifications in {name}"
        raise ValueError(msg)

    for arg in arg_list:
        arg_spec = arg_specs_all[arg]

        if "source" not in arg_spec:
            raise ValueError(f"{arg} missing source field")
        source = arg_spec["source"]
        if not isinstance(source, str):
            raise TypeError(f"{arg}'s source not string ({source})")

        msg = f"Checking argument {arg} of type {source}"
        logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

        if source in TILE_ARGUMENTS_ALL:
            check_tile_specification(arg, arg_spec, logger)
        elif source == EXTERNAL_ARGUMENT:
            check_external_specification(arg, arg_spec, logger)
        elif source == SCRATCH_ARGUMENT:
            check_scratch_specification(arg, arg_spec, logger)
        elif source == GRID_DATA_ARGUMENT:
            check_grid_data_specification(arg, arg_spec,
                                          variable_index_base, logger)
        elif source == THREAD_INDEX_ARGUMENT:
            check_thread_index_specification(arg, arg_spec, logger)
        else:
            msg = f"Unhandled argument {arg} of type {source}"
            raise LogicError(msg)
