def check_partial_tf_specification(spec):
    """
    If this does not raise an error, then the specification is acceptable.

    While this *does* requires that all specifications be provided, it does
    *not* check the values of specifications that are not used.  For
    example, a C++ TF can specify any value for the fortran_source value.

    .. todo::
        * This should also check types
        * Eventually we might have DataPackets sent to CPUs, in which case
          we would want the specification to give the byte alignment as
          well.
        * Allow applications to not specify unnecessary values?  For
          example, do C++ applications always have to specify (with
          whatever value) the Fortran-specific info?
    """
    # ----- ROOT
    expected = {"task_function", "data_item"}
    actual = set(spec)
    if actual != expected:
        msg = f"Invalid root specification keys ({actual})"
        raise ValueError(msg)
    tf_spec = spec["task_function"]
    data_item = spec["data_item"]

    # ----- TASK FUNCTION
    expected = {"language", "processor",
                "cpp_header", "cpp_source",
                "c2f_source", "fortran_source"}
    actual = set(tf_spec)
    if actual != expected:
        msg = f"Invalid TF specification keys ({actual})"
        raise ValueError(msg)
    language = tf_spec["language"]
    processor = tf_spec["processor"]

    if language.lower() not in ["c++", "fortran"]:
        raise ValueError(f"Unsupported TF language ({language})")
    if processor.lower() not in ["cpu", "gpu"]:
        raise ValueError(f"Unsupported target processor ({processor})")
    for each in ["cpp_header", "cpp_source"]:
        if tf_spec[each] == "":
            raise ValueError(f"Empty {each} filename")
    if language.lower() == "fortran":
        for each in ["c2f_source", "fortran_source"]:
            if tf_spec[each] == "":
                raise ValueError(f"Empty {each} filename")

    # ----- DATA ITEM
    expected = {"type", "byte_alignment", "header", "source"}
    actual = set(data_item)
    if actual != expected:
        msg = f"Invalid data item specification keys ({actual})"
        raise ValueError(msg)

    item_type = data_item["type"]
    if item_type.lower() not in ["tilewrapper", "datapacket"]:
        msg = f"Unsupported data item type ({item_type})"
        raise ValueError(msg)

    for each in ["header", "source"]:
        if data_item[each] == "":
            raise ValueError(f"Empty {each} filename")

    if item_type.lower() == "datapacket":
        byte_align = data_item["byte_alignment"]
        if byte_align <= 0:
            raise ValueError("Non-positive byte alignment ({byte_align})")
