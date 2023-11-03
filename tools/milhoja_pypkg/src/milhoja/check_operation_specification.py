import numbers

from .constants import (
    LOG_LEVEL_BASIC_DEBUG,
    ERROR_CHECK_LOG_TAG,
    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT
)
from .LogicError import LogicError
from .AbcLogger import AbcLogger
from .check_grid_specification import check_grid_specification
from .check_subroutine_specification import check_subroutine_specification


def check_operation_specification(spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    .. todo::
        * Once there are lbound and extents parsers in the package, use those
          to error check scratch specification

    :param spec: Contents obtained directly from operation specification file
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    LOG_NAME = ERROR_CHECK_LOG_TAG

    # These must all be lowercase
    VALID_EXTERNAL_TYPES = ["real", "integer", "logical"]
    VALID_SCRATCH_TYPES = ["real"]

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(logger, AbcLogger):
        raise TypeError("Unknown logger type")

    # ----- SPECIFICATION ROOT
    expected = {"format", "grid", "operation"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid operation specification keys ({actual})"
        raise ValueError(msg)

    grid_spec = spec["grid"]
    op_spec = spec["operation"]

    # ----- FORMAT SPECIFICATION
    # The source of the specification is unimportant and was likely used to
    # load the given specification.

    # ----- GRID SPECIFICATION
    msg = "Checking grid subsection specification"
    logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)
    check_grid_specification(grid_spec, logger)

    # ----- OPERATION SPECIFICATION
    msg = "Checking operation subsection specification"
    logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

    # Keys that must always be provided
    if "name" not in op_spec:
        msg = "name key not provided in operation spec"
        raise ValueError(msg)
    elif "variable_index_base" not in op_spec:
        msg = "variable_index_base not provided in operation spec"
        raise ValueError(msg)
    name = op_spec["name"]
    variable_index_base = op_spec["variable_index_base"]

    msg = f"Operation specification under evaluation is {name}"
    logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

    if not isinstance(name, str):
        raise TypeError(f"Operation name not string ({name})")
    elif name == "":
        raise ValueError("Empty operation name")

    if not isinstance(variable_index_base, numbers.Integral):
        msg = f"variable_index_base not integer ({variable_index_base})"
        raise TypeError(msg)
    elif variable_index_base not in [0, 1]:
        msg = f"variable_index_base not 0 or 1 ({variable_index_base})"
        raise ValueError(msg)

    # Optional keys for linking actual arguments across routines
    if EXTERNAL_ARGUMENT in op_spec:
        if len(op_spec[EXTERNAL_ARGUMENT]) == 0:
            raise LogicError(f"Empty {EXTERNAL_ARGUMENT} subsection")

        for variable, var_spec in op_spec[EXTERNAL_ARGUMENT].items():
            expected = {"type", "extents"}
            actual = set(var_spec)
            if actual != expected:
                msg = "Invalid {} variable specification keys ({})"
                raise ValueError(msg.format(EXTERNAL_ARGUMENT, actual))

            var_type = var_spec["type"]
            extents = var_spec["extents"]

            if not isinstance(var_type, str):
                msg = "{} {}'s type is not string ({})"
                msg = msg.format(EXTERNAL_ARGUMENT, variable, var_type)
                raise TypeError(msg)
            elif var_type.lower() not in VALID_EXTERNAL_TYPES:
                msg = "Invalid type ({}) for {} {}"
                msg = msg.format(var_type, EXTERNAL_ARGUMENT, variable)
                raise ValueError(msg)

            if not isinstance(extents, list):
                msg = "{} {}'s extents not a list ({})"
                msg = msg.format(EXTERNAL_ARGUMENT, variable, extents)
                raise TypeError(msg)
            for each in extents:
                if not isinstance(each, numbers.Integral):
                    msg = "Extents of {} {} not integers ({})"
                    msg = msg.format(EXTERNAL_ARGUMENT, variable, extents)
                    raise TypeError(msg)
                elif each <= 0:
                    msg = "Extents of {} {} not positive ({})"
                    msg = msg.format(EXTERNAL_ARGUMENT, variable, extents)
                    raise ValueError(msg)

    if SCRATCH_ARGUMENT in op_spec:
        if len(op_spec[SCRATCH_ARGUMENT]) == 0:
            raise LogicError(f"Empty {SCRATCH_ARGUMENT} subsection")

        for variable, var_spec in op_spec[SCRATCH_ARGUMENT].items():
            expected = {"type", "extents", "lbound"}
            actual = set(var_spec)
            if actual != expected:
                msg = "Invalid {} variable specification keys ({})"
                raise ValueError(msg.format(SCRATCH_ARGUMENT, actual))

            var_type = var_spec["type"]
            extents = var_spec["extents"]
            lbound = var_spec["lbound"]

            if not isinstance(var_type, str):
                msg = "{} {}'s type is not string ({})"
                msg = msg.format(SCRATCH_ARGUMENT, variable, var_type)
                raise TypeError(msg)
            elif var_type.lower() not in VALID_SCRATCH_TYPES:
                msg = "Invalid type ({}) for {} {}"
                msg = msg.format(var_type, SCRATCH_ARGUMENT, variable)
                raise ValueError(msg)

            if not isinstance(extents, str):
                msg = "{} {}'s extents not a string ({})"
                msg = msg.format(SCRATCH_ARGUMENT, variable, extents)
                raise TypeError(msg)

            if not isinstance(lbound, str):
                msg = "{} {}'s lbound not a string ({})"
                msg = msg.format(SCRATCH_ARGUMENT, variable, lbound)
                raise TypeError(msg)

    # ----- SUBROUTINE SPECIFICATIONS
    ignore = {"name", "variable_index_base",
              EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT}
    subroutines_all = set(op_spec).difference(ignore)
    if len(subroutines_all) == 0:
        raise LogicError("No subroutines included in operation")

    for subroutine in subroutines_all:
        msg = f"Checking subroutine {subroutine} specification"
        logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

        check_subroutine_specification(subroutine, op_spec[subroutine],
                                       variable_index_base, logger)

    # ----- CHECK EXTERNAL/SCRATCH LINKAGE
    # Confirm external/scratch argument specifications within subroutines use
    # variables declared at this scope.
    #
    # If we get here, then we can assume that the argument specifications are
    # otherwise acceptable so that we don't have to check before accessing.
    for var_key in [EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT]:
        # Find all low-level variables
        needed = []
        for subroutine in subroutines_all:
            arg_specs_all = op_spec[subroutine]["argument_specifications"]
            for arg, arg_spec in arg_specs_all.items():
                if arg_spec["source"] == var_key:
                    needed.append((arg, arg_spec["name"]))

        if len(needed) > 0:
            if var_key not in op_spec:
                msg = "No high-level {} specifications for mapping to low-level"
                raise ValueError(msg.format(var_key))

            # Keep track of which high-level variables are mapped to low-level
            variables = op_spec[var_key]
            variables = dict(zip(variables, [False] * len(variables)))

            # Confirm all low-level args have a high-level spec
            for arg, name in needed:
                if name not in variables:
                    msg = "Unknown {} variable ({}) for mapping to {}"
                    raise ValueError(msg.format(var_key, name, arg))
                variables[name] = True

            # Check for unused high-level specs
            for name, was_used in variables.items():
                if not was_used:
                    msg = "{} variable {} not used in any subroutine"
                    logger.warn(LOG_NAME, msg.format(var_key, name))
        elif var_key in op_spec:
            variables = set(op_spec[var_key])
            msg = "{} variables not used in any subroutine ({})"
            logger.warn(LOG_NAME, msg.format(var_key, variables))
