import numbers

from .constants import (
    LOG_LEVEL_BASIC_DEBUG,
    ERROR_CHECK_LOG_TAG,
    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT, LBOUND_ARGUMENT
)
from .LogicError import LogicError
from .check_subroutine_specification import check_subroutine_specification


def check_group_specification(group_spec, logger):
    """
    If this does not raise an error, then the specification is acceptable.

    .. todo::
        * Once there are lbound and extents parsers in the package, use those
          to error check scratch specification.  We should not allow for scalar
          scratch arguments.

    :param group_spec: Contents obtained directly from subroutine group
        specification file
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED VALUES
    LOG_NAME = ERROR_CHECK_LOG_TAG

    # These must all be lowercase
    VALID_EXTERNAL_TYPES = ["real", "integer", "logical"]
    VALID_SCRATCH_TYPES = ["real"]

    # ----- ERROR CHECK ARGUMENTS
    if not isinstance(group_spec, dict):
        msg = "Unknown internal subroutine group specification type ({})"
        raise TypeError(msg.format(type(group_spec)))

    # ----- GROUP SPECIFICATION
    msg = "Checking group specification"
    logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

    # Keys that must always be provided
    if "format" not in group_spec:
        msg = "format key not provided in group spec"
        raise ValueError(msg)
    elif "name" not in group_spec:
        msg = "name key not provided in group spec"
        raise ValueError(msg)
    elif "variable_index_base" not in group_spec:
        msg = "variable_index_base not provided in group spec"
        raise ValueError(msg)
    name = group_spec["name"]
    variable_index_base = group_spec["variable_index_base"]

    msg = f"Subroutine group specification under evaluation is {name}"
    logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

    # The source of the specification is unimportant and was likely used to
    # load the given specification.

    if not isinstance(name, str):
        raise TypeError(f"Subroutine group name not string ({name})")
    elif name == "":
        raise ValueError("Empty subroutine group name")

    if not isinstance(variable_index_base, numbers.Integral):
        msg = f"variable_index_base not integer ({variable_index_base})"
        raise TypeError(msg)
    elif variable_index_base not in [0, 1]:
        msg = f"variable_index_base not 0 or 1 ({variable_index_base})"
        raise ValueError(msg)

    # Optional keys for linking actual arguments across routines
    if EXTERNAL_ARGUMENT in group_spec:
        if len(group_spec[EXTERNAL_ARGUMENT]) == 0:
            raise LogicError(f"Empty {EXTERNAL_ARGUMENT} subsection")

        for variable, var_spec in group_spec[EXTERNAL_ARGUMENT].items():
            if not variable.startswith("_"):
                msg = "Global {} variable {} does not start with '_'"
                raise ValueError(msg.format(EXTERNAL_ARGUMENT, variable))

            expected = {"type", "extents"}
            actual = set(var_spec)
            for key in expected:
                if key not in actual:
                    msg = f"{key} not in {variable}'s specification ({actual})"
                    raise ValueError(msg)

            # Allow applications to store in JSON information about where
            # external variables come from.  Permit each application to store
            # this information however they see fit by simply ignoring this
            # optional field.
            optionals = actual.difference(expected)
            if (len(optionals) > 0) and (optionals != {"application_specific"}):
                msg = f"Invalid extra external keys ({optionals})"
                raise ValueError(msg)

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

            if not isinstance(extents, str):
                msg = "{} {}'s extents not string ({})"
                msg = msg.format(EXTERNAL_ARGUMENT, variable, extents)
                raise TypeError(msg)

    if SCRATCH_ARGUMENT in group_spec:
        if len(group_spec[SCRATCH_ARGUMENT]) == 0:
            raise LogicError(f"Empty {SCRATCH_ARGUMENT} subsection")

        for variable, var_spec in group_spec[SCRATCH_ARGUMENT].items():
            if not variable.startswith("_"):
                msg = "Global {} variable {} does not start with '_'"
                raise ValueError(msg.format(SCRATCH_ARGUMENT, variable))

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
    ignore = {"format", "name", "variable_index_base",
              EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT}
    subroutines_all = set(group_spec).difference(ignore)
    if len(subroutines_all) == 0:
        raise LogicError("No subroutines included in group")

    for subroutine in subroutines_all:
        msg = f"Checking subroutine {subroutine} specification"
        logger.log(LOG_NAME, msg, LOG_LEVEL_BASIC_DEBUG)

        check_subroutine_specification(subroutine, group_spec[subroutine],
                                       variable_index_base, logger)

    # ----- CHECK EXTERNAL/SCRATCH LINKAGE
    # Confirm external/scratch argument specifications within subroutines use
    # variables declared at this scope.
    #
    # If we get here, then we can assume that the argument specifications are
    # otherwise acceptable so that we don't have to check before accessing.
    OUTER_ARGS = [EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT]
    for var_key in OUTER_ARGS:
        # Find all low-level variables
        needed = []
        for subroutine in subroutines_all:
            arg_specs_all = group_spec[subroutine]["argument_specifications"]
            for arg, arg_spec in arg_specs_all.items():
                if arg_spec["source"] == var_key:
                    needed.append((arg, arg_spec["name"]))

        if len(needed) > 0:
            if var_key not in group_spec:
                msg = "No high-level {} specifications for mapping to low-level"
                raise ValueError(msg.format(var_key))

            # Keep track of which high-level variables are mapped to low-level
            variables = group_spec[var_key]
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
        elif var_key in group_spec:
            variables = set(group_spec[var_key])
            msg = "{} variables not used in any subroutine ({})"
            logger.warn(LOG_NAME, msg.format(var_key, variables))

    # ----- CHECK LBOUND ARRAYS ARE ACTUALLY ARRAYS
    for subroutine in subroutines_all:
        arg_specs_all = group_spec[subroutine]["argument_specifications"]
        for arg, arg_spec in arg_specs_all.items():
            if arg_spec["source"] == LBOUND_ARGUMENT:
                array_name = arg_spec["array"]
                array_spec = arg_specs_all[array_name]
                array_source = array_spec["source"]
                # The lbounds have already been checked to confirm that their
                # arrays are of an acceptable type.  If we assume that scratch
                # arguments must always be arrays, then all types other than
                # external must be arrays.
                if array_source == EXTERNAL_ARGUMENT:
                    outer_name = array_spec["name"]
                    extents = group_spec[array_source][outer_name]["extents"]
                    if extents == "()":
                        msg = "lbound {}'s array is scalar external variable {}"
                        raise ValueError(msg.format(arg, outer_name))
