import numbers


def check_partial_tf_specification(spec):
    """
    If this does not raise an error, then the specification is acceptable.

    While this **does** require that all specifications be provided, it does
    **not** check the values of specifications that are not used.  For
    example, a C++ TF can specify any value for the fortran_source value
    without this function raising an error.

    .. todo::
        * Eventually we might have DataPackets sent to CPUs, in which case
          we would want the specification to give the byte alignment as
          well.
    """
    # ----- ROOT
    expected = {"task_function", "data_item"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid root specification keys ({actual})"
        raise ValueError(msg)

    # ----- TASK FUNCTION
    tf_spec = spec["task_function"]
    expected = {"language", "processor",
                "cpp_header", "cpp_source",
                "c2f_source", "fortran_source",
                "computation_offloading",
                "variable_index_base"}
    actual = set(tf_spec)
    if actual != expected:
        msg = f"Invalid TF specification keys ({actual})"
        raise ValueError(msg)

    language = tf_spec["language"]
    if not isinstance(language, str):
        raise TypeError(f"language not string ({language})")
    elif language.lower() not in ["c++", "fortran"]:
        raise ValueError(f"Unsupported TF language ({language})")

    processor = tf_spec["processor"]
    if not isinstance(processor, str):
        raise TypeError(f"processor not string ({processor})")
    elif processor.lower() not in ["cpu", "gpu"]:
        raise ValueError(f"Unsupported target processor ({processor})")

    for each in ["cpp_header", "cpp_source"]:
        fname = tf_spec[each]
        if not isinstance(fname, str):
            raise TypeError(f"{each} not string ({fname})")
        elif fname == "":
            raise ValueError(f"Empty {each} filename")

    if language.lower() == "fortran":
        for each in ["c2f_source", "fortran_source"]:
            fname = tf_spec[each]
            if not isinstance(fname, str):
                raise TypeError(f"{each} not string ({fname})")
            elif fname == "":
                raise ValueError(f"Empty {each} filename")

    offloading = tf_spec["computation_offloading"]
    if not isinstance(offloading, str):
        raise TypeError(f"computation_offloading not string ({offloading})")
    elif (processor.lower() == "cpu") and (offloading != ""):
        raise ValueError("No computation offloading for CPU task functions")
    elif (processor.lower() == "gpu") and (offloading.lower() not in ("openacc", "openmp")):
        raise ValueError("Only OpenACC or OpenMP computation offloading with GPU")

    # ----- DATA ITEM
    data_item = spec["data_item"]
    expected = {"type", "byte_alignment", "header", "source", "module"}
    actual = set(data_item)
    if actual != expected:
        msg = f"Invalid data item specification keys ({actual})"
        raise ValueError(msg)

    item_type = data_item["type"]
    if not isinstance(item_type, str):
        raise TypeError(f"Data item type not string ({item_type})")
    elif item_type.lower() not in ["tilewrapper", "datapacket"]:
        msg = f"Unsupported data item type ({item_type})"
        raise ValueError(msg)

    if item_type.lower() == "datapacket":
        byte_align = data_item["byte_alignment"]
        if not isinstance(byte_align, numbers.Integral):
            msg = f"Data item byte alignment not integer ({byte_align})"
            raise TypeError(msg)
        elif byte_align <= 0:
            raise ValueError(f"Non-positive byte alignment ({byte_align})")

    for each in ["header", "source"]:
        fname = data_item[each]
        if not isinstance(fname, str):
            raise TypeError(f"Data item {each} not string ({fname})")
        elif fname == "":
            raise ValueError(f"Empty data item {each} filename")

    fname = data_item["module"]
    if not isinstance(fname, str):
        raise TypeError(f"Data item module {each} not string ({fname})")
