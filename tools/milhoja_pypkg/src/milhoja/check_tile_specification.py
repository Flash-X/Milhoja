from .constants import (
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    TILE_ARGUMENTS_ALL
)
from .LogicError import LogicError
from .AbcLogger import AbcLogger


def check_tile_specification(arg, spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    :param arg: Name of argument
    :param spec: Argument specification obtained directly from subroutine
        specification
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    # LOG_NAME = ERROR_CHECK_LOG_TAG

    VALID_AXIS = ["i", "j", "k"]
    VALID_EDGE = ["left", "center", "right"]
    VALID_LO = [TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]
    VALID_HI = [TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    if "source" not in spec:
        raise ValueError(f"{arg} missing source field")
    source = spec["source"]
    if not isinstance(source, str):
        raise TypeError(f"{arg}'s source not string ({source})")
    elif source not in TILE_ARGUMENTS_ALL:
        msg = f"{arg}'s specification not for tile argument ({source})"
        raise LogicError(msg)

    # ----- TILE METADATA SPECIFICATIONS
    singletons = [TILE_GRID_INDEX_ARGUMENT,
                  TILE_LEVEL_ARGUMENT,
                  TILE_DELTAS_ARGUMENT,
                  TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
                  TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
                  TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT]
    if source in singletons:
        if len(spec) != 1:
            keys = list(spec.keys())
            msg = f"Too many fields for {arg}'s specification ({keys})"
            raise ValueError(msg)
    elif source == TILE_COORDINATES_ARGUMENT:
        expected = {"source", "axis", "edge", "lo", "hi"}
        actual = set(spec)
        if actual != expected:
            msg = f"Invalid set of {source} specification keys ({actual})"
            raise ValueError(msg)
        axis = spec["axis"]
        edge = spec["edge"]
        lo = spec["lo"]
        hi = spec["hi"]

        if not isinstance(axis, str):
            msg = f"{arg}'s axis value not string ({axis})"
            raise TypeError(msg)
        elif axis.lower() not in VALID_AXIS:
            msg = f"Invalid {source} axis value ({axis}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(edge, str):
            msg = f"{arg}'s edge value not string ({edge})"
            raise TypeError(msg)
        elif edge.lower() not in VALID_EDGE:
            msg = f"Invalid {source} edge value ({edge}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(lo, str):
            msg = f"{arg}'s lo value not string ({lo})"
            raise TypeError(msg)
        elif lo not in VALID_LO:
            msg = f"Invalid {source} lo type ({lo}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(hi, str):
            msg = f"{arg}'s hi value not string ({hi})"
            raise TypeError(msg)
        elif hi not in VALID_HI:
            msg = f"Invalid {source} hi type ({hi}) for {arg}"
            raise ValueError(msg)
    elif source == TILE_FACE_AREAS_ARGUMENT:
        expected = {"source", "axis", "lo", "hi"}
        actual = set(spec)
        if actual != expected:
            msg = f"Invalid set of {source} specification keys ({actual})"
            raise ValueError(msg)
        axis = spec["axis"]
        lo = spec["lo"]
        hi = spec["hi"]

        if not isinstance(axis, str):
            msg = f"{arg}'s axis value not string ({axis})"
            raise TypeError(msg)
        elif axis.lower() not in VALID_AXIS:
            msg = f"Invalid {source} axis value ({axis}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(lo, str):
            msg = f"{arg}'s lo value not string ({lo})"
            raise TypeError(msg)
        elif lo not in VALID_LO:
            msg = f"Invalid {source} lo type ({lo}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(hi, str):
            msg = f"{arg}'s hi value not string ({hi})"
            raise TypeError(msg)
        elif hi not in VALID_HI:
            msg = f"Invalid {source} hi type ({hi}) for {arg}"
            raise ValueError(msg)
    elif source == TILE_CELL_VOLUMES_ARGUMENT:
        expected = {"source", "lo", "hi"}
        actual = set(spec)
        if actual != expected:
            msg = f"Invalid set of {source} specification keys ({actual})"
            raise ValueError(msg)
        lo = spec["lo"]
        hi = spec["hi"]

        if not isinstance(lo, str):
            msg = f"{arg}'s lo value not string ({lo})"
            raise TypeError(msg)
        elif lo not in VALID_LO:
            msg = f"Invalid {source} lo type ({lo}) for {arg}"
            raise ValueError(msg)
        elif not isinstance(hi, str):
            msg = f"{arg}'s hi value not string ({hi})"
            raise TypeError(msg)
        elif hi not in VALID_HI:
            msg = f"Invalid {source} hi type ({hi}) for {arg}"
            raise ValueError(msg)
    else:
        msg = f"Tile argument {arg} with source {source} not handled"
        raise LogicError(msg)
