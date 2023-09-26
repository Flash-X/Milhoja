def generate_tile_metadata_extraction(specs_all, tile_desc):
    """
    """
    print(specs_all)

    code = []

    # ----- ADD TILEMETADATA NEEDED INTERNALLY
    # Some tile metadata can only be accessed using other metadata

    for arg, spec in specs_all.items():
        metadata_types = [spec["source"].lower()
                          for _, spec in specs_all.items()]
        if spec["source"].lower() == "tile_coordinates":
            for requirement in ["tile_level"]:
                if requirement not in metadata_types:
                    key = "__MH_INTERNAL_level"
                    if key in specs_all:
                        raise ValueError("Internal level logic error")
                    spec[key] = {"source": "tile_level"}

    # ----- EXTRACT INDEPENDENT METADATA
    print(specs_all)
    for arg, spec in specs_all.items():
        source = spec["source"].lower()

        arg_type = None
        getter = None
        if source == "tile_gridindex":
            arg_type, getter = "const int", "gridIndex"
        elif source == "tile_level":
            arg_type, getter = "const unsigned int", "level"
        elif source == "tile_lo":
            arg_type, getter = "const milhoja::IntVect", "lo"
        elif source == "tile_hi":
            arg_type, getter = "const milhoja::IntVect", "hi"
        elif source == "tile_lbound":
            arg_type, getter = "const milhoja::IntVect", "loGC"
        elif source == "tile_ubound":
            arg_type, getter = "const milhoja::IntVect", "hiGC"
        elif source == "tile_deltas":
            arg_type, getter = "const milhoja::RealVect", "deltas"

        if arg_type and getter:
            line = f"{arg_type} {arg} = {tile_desc}->{getter}();"
            code.append(line)
    code.append("")

    # ----- EXTRACT DEPENDENT METADATA
    for arg, spec in specs_all.items():
        if spec["source"].lower() == "tile_coordinates":
            line = "hello"

    return code
