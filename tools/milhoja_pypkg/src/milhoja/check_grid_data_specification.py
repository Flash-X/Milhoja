import numbers

from .constants import GRID_DATA_ARGUMENT
from .LogicError import LogicError
from .AbcLogger import AbcLogger


def check_grid_data_specification(arg, spec, variable_index_base, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    .. todo::
        * Will this also work for task function specifications?  It would need
          to allow only R/RW/W or variable_in/_out.

    :param arg: Name of argument
    :param spec: Argument specification obtained directly from subroutine
        specification
    :param variable_index_base: Minimum value in variable index sets
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    # LOG_NAME = ERROR_CHECK_LOG_TAG

    VALID_SPACES = ["center", "fluxx", "fluxy", "fluxz"]
    VALID_VAR_INFO = ["R", "RW", "W"]

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    if "source" not in spec:
        raise ValueError(f"{arg} missing source field")
    source = spec["source"]
    if not isinstance(source, str):
        raise TypeError(f"{arg}'s source not string ({source})")
    elif source != GRID_DATA_ARGUMENT:
        msg = f"{arg}'s specification not for grid data argument ({source})"
        raise LogicError(msg)

    # ----- GRID-DATA SPECIFICATIONS
    if "structure_index" not in spec:
        msg = f"{arg} missing structure_index information"
        raise ValueError(msg)
    s_idx = spec["structure_index"]

    if not isinstance(s_idx, list):
        msg = f"{arg}'s structure_index is not list ({s_idx})"
        raise TypeError(msg)
    elif len(s_idx) != 2:
        msg = f"Incorrect size of {arg}'s structure index ({s_idx})"
        raise ValueError(msg)
    space, idx = s_idx

    if not isinstance(space, str):
        msg = f"{arg}'s index space is not string ({s_idx})"
        raise TypeError(msg)
    elif space.lower() not in VALID_SPACES:
        msg = f"Invalid index space for {arg} ({s_idx})"
        raise ValueError(msg)
    elif not isinstance(idx, numbers.Integral):
        msg = f"{arg}'s data structure index not integer ({s_idx})"
        raise TypeError(msg)
    elif idx != 1:
        msg = f"Invalid data structure index for {arg} ({s_idx})"
        raise ValueError(msg)

    ignore = {"source", "structure_index"}
    var_specs_all = set(spec).difference(ignore)
    if not var_specs_all:
        msg = f"No variable access patterns for grid data {arg}"
        raise ValueError(msg)
    unknown = set(var_specs_all).difference(VALID_VAR_INFO)
    if unknown != set():
        msg = f"Invalid variable access patterns for {arg} ({unknown})"
        raise ValueError(msg)

    # Verify each access pattern first
    n_specs = len(var_specs_all)
    var_specs_all = list(var_specs_all)
    for i, access in enumerate(var_specs_all):
        pattern = spec[access]
        if not isinstance(pattern, list):
            msg = f"{arg}'s {access} access pattern not list ({pattern})"
            raise TypeError(msg)
        elif not pattern:
            msg = f"Empty {access} access pattern for {arg}"
            raise ValueError(msg)
        for idx in pattern:
            if not isinstance(idx, numbers.Integral):
                msg = f"{arg} {access} access not integer ({pattern})"
                raise TypeError(msg)
            elif idx < variable_index_base:
                msg = "{}'s {} access ({}) less than variable_index_base"
                raise ValueError(msg.format(arg, access, idx))
        if len(pattern) > len(set(pattern)):
            msg = f"Repeated variable in {arg} for {access} access ({pattern})"
            raise LogicError(msg)

    # No variable can have more than one access pattern
    for i, access_i in enumerate(var_specs_all):
        spec_i = set(spec[access_i])
        for j in range(i+1, n_specs):
            access_j = var_specs_all[j]
            spec_j = set(spec[access_j])
            overlap = spec_i.intersection(spec_j)
            if overlap != set():
                msg = f"{arg} variable assigned to two or more accesses"
                raise ValueError(msg)
