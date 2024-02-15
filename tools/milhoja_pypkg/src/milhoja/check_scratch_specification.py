from .constants import SCRATCH_ARGUMENT
from .LogicError import LogicError
from .AbcLogger import AbcLogger


def check_scratch_specification(arg, spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    At this level, the test cannot know if the name of the associated scratch
    variable is correct.  That must be checked at a higher level.

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
    elif source != SCRATCH_ARGUMENT:
        msg = f"{arg}'s specification not for scratch argument ({source})"
        raise LogicError(msg)

    # ----- SCRATCH SPECIFICATIONS
    expected = {"source", "name"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid scratch specification keys for {arg} ({actual})"
        raise ValueError(msg)

    var_name = spec["name"]
    if not isinstance(var_name, str):
        msg = f"{arg}'s name is not string ({var_name})"
        raise TypeError(msg)
