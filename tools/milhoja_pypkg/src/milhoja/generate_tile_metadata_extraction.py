def generate_tile_metadata_extraction(task_function, tile_desc):
    """
    """
    code = []

    args_all = task_function.dummy_arguments
    metadata_all = task_function.tile_metadata_arguments

    # ----- ADD TILEMETADATA NEEDED INTERNALLY
    # Some tile metadata can only be accessed using other metadata.
    # Add dependent metadata if not already in list.
    #
    # TODO: Should we have a function that generates variable names since the
    # same MH_INTERNAL_* variables used here are also used in another code
    # generator.
    internal = {}
    for arg in args_all:
        spec = task_function.argument_specification(arg)

        dependents = ["tile_coordinates", "tile_faceAreas", "tile_cellVolumes"]
        if spec["source"] in dependents:
            if "tile_level" not in metadata_all:
                variable = "MH_INTERNAL_level"
                if variable not in internal:
                    internal[variable] = {"source": "tile_level"}
                    metadata_all["tile_level"] = [variable]

            for point in ["lo", "hi"]:
                key = spec[point].strip().lower()
                if key not in metadata_all:
                    assert key.startswith("tile_")
                    short = key.replace("tile_", "")
                    assert short in ["lo", "hi", "lbound", "ubound"]

                    variable  = f"MH_INTERNAL_{short}"
                    if variable not in internal:
                        internal[variable] = {"source": key}
                        metadata_all[key] = [variable]

    # ----- EXTRACT INDEPENDENT METADATA
    # TODO: This is for CPU/C++
    order = [("tile_gridIndex", "const int", "gridIndex"),
             ("tile_level", "const unsigned int", "level"),
             ("tile_lo", "const milhoja::IntVect", "lo"),
             ("tile_hi", "const milhoja::IntVect", "hi"),
             ("tile_lbound", "const milhoja::IntVect", "loGC"),
             ("tile_ubound", "const milhoja::IntVect", "hiGC"),
             ("tile_deltas", "const milhoja::RealVect", "deltas")]
    for key, arg_type, getter in order:
        if key in metadata_all:
            arg = metadata_all[key]
            assert len(arg) == 1
            arg = arg[0]

            line = f"{arg_type}   {arg} = {tile_desc}->{getter}();"
            code.append(line)

    # ----- CREATE THREAD-PRIVATE INTERNAL SCRATCH
    if "tile_cellVolumes" in metadata_all:
        arg_list = metadata_all["tile_cellVolumes"]
        assert len(arg_list) == 1
        arg = arg_list[0]
        wrapper = f"Tile_{task_function.name}"
        code += [
            f"milhoja::Real*   MH_INTERNAL_cellVolumes_ptr =",
            f"\tstatic_cast<milhoja::Real*>({wrapper}::MH_INTERNAL_cellVolumes_)",
            f"\t+ {wrapper}::MH_INTERNAL_CELLVOLUMES_SIZE_ * threadId;"
        ]

    # ----- EXTRACT DEPENDENT METADATA
    axis_mh = {"i": "milhoja::Axis::I",
               "j": "milhoja::Axis::J",
               "k": "milhoja::Axis::K"}
    edge_mh = {"center": "milhoja::Edge::Center"}

    if "tile_coordinates" in metadata_all:
        arg_list = metadata_all["tile_coordinates"]
        assert len(arg_list) <= 3
        for arg in arg_list:
            spec = task_function.argument_specification(arg)
            axis = axis_mh[spec["axis"].lower()]
            edge = edge_mh[spec["edge"].lower()]
            level = metadata_all["tile_level"][0]
            lo = metadata_all[spec["lo"]][0]
            hi  = metadata_all[spec["hi"]][0]
            code += [
                f"const milhoja::FArray1D  {arg} =",
                "\tmilhoja::Grid::instance().getCellCoords(",
                f"\t\t{axis},",
                f"\t\t{edge},",
                f"\t\t{level},",
                f"\t\t{lo}, {hi});"
            ]

    if "tile_faceAreas" in metadata_all:
        raise NotImplementedError("No test case yet for face areas")

    if "tile_cellVolumes" in metadata_all:
        arg_list = metadata_all["tile_cellVolumes"]
        assert len(arg_list) == 1
        arg = arg_list[0]
        spec = task_function.argument_specification(arg)
        level = metadata_all["tile_level"][0]
        lo = metadata_all[spec["lo"]][0]
        hi  = metadata_all[spec["hi"]][0]
        code += [
            f"Grid::instance().fillCellVolumes(",
            f"\t{level},",
            f"\t{lo},",
            f"\t{hi},",
             "\tMH_INTERNAL_cellVolumes_ptr);",
            f"const milhoja::FArray3D  {arg}{{",
             "\t\tMH_INTERNAL_cellVolumes_ptr,",
            f"\t\t{lo},",
            f"\t\t{hi}}};"
        ]

    return code
