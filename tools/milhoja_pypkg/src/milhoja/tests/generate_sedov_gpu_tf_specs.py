import json

from milhoja import TaskFunctionAssembler


def generate_sedov_gpu_tf_specs(dimension, block_size,
                                op_spec_path, destination, overwrite,
                                logger):
    """
    Use the partial operation specifications located in the given path to fill
    in problem-specific operation specifications and construct the full
    specification JSON file for each task function needed to solve the given
    Sedov test problem.

    :param dimension: Dimension of problem's domain
    :param block_size: (nxb, nyb, nzb) where n[xyz]b is N cells in each block
        along associated dimension of problem's domain
    :param op_spec_path: Path to location of Sedov problem's partial operation
        specification JSON files
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing JSON files in destination will be overwritten
        if True
    :param logger: Logger derived from milhoja.AbcLogger
    """
    # ----- HARDCODED
    TF_CALL_GRAPH = [
        "Hydro_computeSoundSpeedHll_gpu_oacc",
        [
            "Hydro_computeFluxesHll_X_gpu_oacc",
            "Hydro_computeFluxesHll_Y_gpu_oacc",
            "Hydro_computeFluxesHll_Z_gpu_oacc"
        ],
        "Hydro_updateSolutionHll_gpu_oacc"
    ]

    GRID_SPEC = {
        "dimension": dimension,
        "nxb": block_size[0],
        "nyb": block_size[1],
        "nzb": block_size[2],
        "nguardcells": 1
    }

    PARTIAL_TF_SPEC = {
        "task_function": {
            "language":       "Fortran",
            "processor":      "GPU",
            "cpp_header":     "gpu_tf_hydro_Cpp2C.h",
            "cpp_source":     "gpu_tf_hydro_Cpp2C.cpp",
            "c2f_source":     "gpu_tf_hydro_C2F.F90",
            "fortran_source": "gpu_tf_hydro.F90"
        },
        "data_item": {
            "type":           "DataPacket",
            "byte_alignment": 16,
            "header":         "DataPacket_gpu_tf_hydro.h",
            "source":         "DataPacket_gpu_tf_hydro.cpp"
        }
    }

    # ----- ADJUST OPERATION SPECIFICATION TO SPECIFIC PROBLEM
    group_json = op_spec_path.joinpath("Hydro_op1_Fortran.json")
    if not group_json.is_file():
        msg = f"{group_json} does not exist or is not a file"
        raise ValueError(msg)
    with open(group_json, "r") as fptr:
        group_spec = json.load(fptr)

    # Add in Grid Spec
    assert "grid" not in group_spec
    group_spec["grid"] = GRID_SPEC

    # Scratch extents change with dimension
    sz_x = block_size[0] + 2
    sz_y = block_size[1] + 2 if dimension >= 2 else 1
    sz_z = block_size[2] + 2 if dimension == 3 else 1
    extents = f"({sz_x}, {sz_y}, {sz_z})"
    group_spec["operation"]["scratch"]["_auxC"]["extents"] = extents

    sz_x = 1
    sz_y = 1 if dimension >= 2 else 0
    sz_z = 1 if dimension == 3 else 0
    lbound = f"(tile_lo) - ({sz_x}, {sz_y}, {sz_z})"
    group_spec["operation"]["scratch"]["_auxC"]["lbound"] = lbound

    for i, each in enumerate(["_flX", "_flY", "_flZ"]):
        fl_size = block_size.copy()
        fl_size[i] += 1

        sz_x = fl_size[0] if i < dimension else 1
        sz_y = fl_size[1] if i < dimension else 1
        sz_z = fl_size[2] if i < dimension else 1
        n_flux = 5 if i < dimension else 1

        extents = f"({sz_x}, {sz_y}, {sz_z}, {n_flux})"
        group_spec["operation"]["scratch"][each]["extents"] = extents

        lbound = "(tile_lo, 1)" if i < dimension else "(1, 1, 1, 1)"
        group_spec["operation"]["scratch"][each]["lbound"] = lbound

    # Dump final operation specification for immediate use
    filename = f"Hydro_op1_Fortran_{dimension}D.json"
    group_json = destination.joinpath(filename)
    if (not overwrite) and group_json.exists():
        raise ValueError(f"{group_json} already exists")
    with open(group_json, "w") as fptr:
        json.dump(group_spec, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    # ----- DUMP PARTIAL TF SPECIFICATION
    filename = f"gpu_tf_hydro_partial_{dimension}D.json"
    partial_tf_spec_json = destination.joinpath(filename)
    if (not overwrite) and partial_tf_spec_json.exists():
        msg = f"{partial_tf_spec_json} already exists"
        raise ValueError(msg)
    with open(partial_tf_spec_json, "w") as fptr:
        json.dump(PARTIAL_TF_SPEC, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    # ----- GENERATE TASK FUNCTION SPECIFICATION JSON
    full_tf_spec = destination.joinpath(f"gpu_tf_hydro_{dimension}D.json")
    assembler = TaskFunctionAssembler.from_milhoja_json(
                    "gpu_tf_hydro", TF_CALL_GRAPH, [group_json],
                    logger
                )
    assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

    return full_tf_spec
