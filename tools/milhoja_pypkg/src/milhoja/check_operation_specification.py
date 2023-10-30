from .check_grid_specification import check_grid_specification


def check_operation_specification(spec):
    """
    If this does not raise an error, then the specification is acceptable.

    .. todo::
        * Take operation name as parameter for including in error messages
        * Once there are lbound and extents parsers in the package, use those
          to error check scratch specification

    :param spec: Contents obtained directly from operation specification file
    """
    # ----- SPECIFICATION ROOT
    expected = {"format", "grid", "operation"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid operation specification keys ({actual})"
        raise ValueError(msg)

    grid_spec = spec["grid"]
    op_spec = spec["operation"]

    # ----- FORMAT SPECIFICATION
    # The source of the specification is unimportant

    # ----- GRID SPECIFICATION
    check_grid_specification(grid_spec)

    # ----- OPERATION SPECIFICATION
    # Keys that must always be provided
    if "name" not in op_spec:
        msg = "Name of operation not provided in operation spec"
        raise ValueError(msg)
    elif "variable_index_base" not in op_spec:
        msg = "variable_index_base not provided in operation spec"
        raise ValueError(msg)

    # Optional keys for linking actual arguments across routines
    if "external" in op_spec:
        for variable, var_spec in op_spec["external"].items():
            expected = {"type", "extents"}
            actual = set(var_spec)
            if actual != expected:
                msg = f"Invalid external variable specification keys ({actual})"
                raise ValueError(msg)

            var_type = var_spec["type"]
            extents = var_spec["extents"]

            if not isinstance(var_type, str):
                msg = f"External {variable}'s type is not string ({var_type})"
                raise TypeError(msg)
            elif var_type.lower() not in ["real"]:
                msg = f"Invalid type ({var_type}) for external {variable}"
                raise ValueError(msg)

            if not isinstance(extents, list):
                msg = f"External {variable}'s extents not a list ({extents})"
                raise TypeError(msg)
            elif extents:
                msg = f"External {variable} must have extents set to []"
                raise ValueError(msg)

    if "scratch" in op_spec:
        for variable, var_spec in op_spec["scratch"].items():
            expected = {"type", "extents", "lbound"}
            actual = set(var_spec)
            if actual != expected:
                msg = f"Invalid scratch variable specification keys ({actual})"
                raise ValueError(msg)

            var_type = var_spec["type"]
            extents = var_spec["extents"]
            lbound = var_spec["lbound"]

            if not isinstance(var_type, str):
                msg = f"Scratch {variable}'s type is not string ({var_type})"
                raise TypeError(msg)
            elif var_type.lower() not in ["real"]:
                msg = f"Invalid type ({var_type}) for scratch {variable}"
                raise ValueError(msg)

            if not isinstance(extents, str):
                msg = f"Scratch {variable}'s extents not a string ({extents})"
                raise TypeError(msg)

            if not isinstance(lbound, str):
                msg = f"Scratch {variable}'s lbound not a string ({lbound})"
                raise TypeError(msg)
