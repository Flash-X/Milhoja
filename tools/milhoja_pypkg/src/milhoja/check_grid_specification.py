import numbers

from .AbcLogger import AbcLogger


def check_grid_specification(spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    .. todo::
        * Will this also work for task function specifications?

    :param spec: Grid specification obtained directly from operation
        specification
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    # LOG_NAME = ERROR_CHECK_LOG_TAG

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(spec, dict):
        msg = "Unknown grid specification type ({})"
        raise TypeError(msg.format(type(spec)))
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    # ----- GRID SPECIFICATION
    expected = {"dimension", "nxb", "nyb", "nzb", "nguardcells"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid set of grid specification keys ({actual})"
        raise ValueError(msg)

    dimension = spec["dimension"]
    if not isinstance(dimension, numbers.Integral):
        raise TypeError(f"Dimension is not integer ({dimension})")
    if dimension not in [1, 2, 3]:
        msg = f"Invalid grid dimension ({dimension})"
        raise ValueError(msg)

    nxb = spec["nxb"]
    if not isinstance(nxb, numbers.Integral):
        raise TypeError(f"NXB is not integer ({nxb})")
    elif nxb <= 0:
        raise ValueError(f"Non-positive NXB ({nxb})")

    nyb = spec["nyb"]
    if not isinstance(nyb, numbers.Integral):
        raise TypeError(f"NYB is not integer ({nyb})")
    elif nyb <= 0:
        raise ValueError(f"Non-positive NYB ({nyb})")
    elif (dimension == 1) and (nyb != 1):
        raise ValueError("nyb > 1 for 1D problem")

    nzb = spec["nzb"]
    if not isinstance(nzb, numbers.Integral):
        raise TypeError(f"NZB is not integer ({nzb})")
    elif nzb <= 0:
        raise ValueError(f"Non-positive NZB ({nzb})")
    elif (dimension < 3) and (nzb != 1):
        raise ValueError(f"nzb > 1 for {dimension}D problem")

    n_gc = spec["nguardcells"]
    if not isinstance(n_gc, numbers.Integral):
        raise TypeError(f"nguardcells is not integer ({n_gc})")
    elif n_gc < 0:
        raise ValueError(f"Negative N guardcells ({n_gc})")
