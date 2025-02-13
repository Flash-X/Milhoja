from .constants import VERBATIM_ARGUMENT
from .LogicError import LogicError
from .AbcLogger import AbcLogger


def check_verbatim_specification(arg, spec, logger):
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
    elif source != VERBATIM_ARGUMENT:
        msg = f"{arg}'s specification not for verbatim argument ({source})"
        raise LogicError(msg)

    # ----- VERBATIM SPECIFICATIONS
    expected = {"source", "application_specific"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid verbatim specification keys for {arg} ({spec})"
        raise ValueError(msg)
    subspec = spec["application_specific"]

    expected = {"kind", "value"}
    actual = set(subspec)
    if actual != expected:
        msg = f"Invalid verbatim \"application_specific\" keys for {arg} ({subspec})"
        raise ValueError(msg)

    verbatim_kind = subspec["kind"]
    if not isinstance(verbatim_kind, str):
        raise TypeError(f"{arg}'s verbatim kind not string ({verbatim_kind})")
    if verbatim_kind != "literal":
        raise TypeError(f"{arg}'s verbatim kind not recognized ({verbatim_kind})")

    verbatim_value = subspec["value"]
    if not isinstance(verbatim_value, str):
        raise TypeError(f"{arg}'s verbatim value not string ({verbatim_value})")
