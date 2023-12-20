from .constants import THREAD_INDEX_ARGUMENT
from .LogicError import LogicError
from .AbcLogger import AbcLogger


def check_thread_index_specification(arg, spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    :param arg: Name of argument
    :param spec: Argument specification obtained directly from
        subroutine specification
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    # LOG_NAME = ERROR_CHECK_LOG_TAG

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    if "source" not in spec:
        raise ValueError(f"{arg} missing source field")
    source = spec["source"]
    if not isinstance(source, str):
        raise TypeError(f"{arg}'s source not string ({source})")
    elif source != THREAD_INDEX_ARGUMENT:
        msg = f"{arg}'s specification not for thread index argument ({source})"
        raise LogicError(msg)

    # ----- THREAD INDEX SPECIFICATIONS
    if len(spec) != 1:
        keys = list(spec.keys())
        msg = f"Too many fields for {arg} thread index specification ({keys})"
        raise ValueError(msg)
