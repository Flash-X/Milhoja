import json

from milhoja import (
    SubroutineGroup,
    TaskFunctionAssembler
)


def generate_flashx_cpu_tf_specs(dimension, block_size,
                                 group_spec_path, destination, overwrite,
                                 logger):
    """
    Use the partial subroutine group specifications located in the given path
    to fill in problem-specific subroutine group specifications and construct
    the full specification JSON file for each task function needed to solve the
    given Flash-X/Sedov test problem using the CPU only.

    :param dimension: Dimension of problem's domain
    :param block_size: (nxb, nyb, nzb) where n[xyz]b is N cells in each block
        along associated dimension of problem's domain
    :param group_spec_path: Path to location of Sedov problem's partial
        subroutine group specification JSON files
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing JSON files in destination will be overwritten
        if True
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED
    if dimension == 1:
        TF_CALL_GRAPH = [
            "Hydro_computeSoundSpeedHll_block_cpu",
            "Hydro_computeFluxesHll_X_block_cpu",
            "Hydro_updateSolutionHll_block_cpu",
            "Eos_wrapped"
        ]
    elif dimension == 2:
        TF_CALL_GRAPH = [
            "Hydro_computeSoundSpeedHll_block_cpu",
            "Hydro_computeFluxesHll_X_block_cpu",
            "Hydro_computeFluxesHll_Y_block_cpu",
            "Hydro_updateSolutionHll_block_cpu",
            "Eos_wrapped"
        ]
    elif dimension == 3:
        TF_CALL_GRAPH = [
            "Hydro_computeSoundSpeedHll_block_cpu",
            "Hydro_computeFluxesHll_X_block_cpu",
            "Hydro_computeFluxesHll_Y_block_cpu",
            "Hydro_computeFluxesHll_Z_block_cpu",
            "Hydro_updateSolutionHll_block_cpu",
            "Eos_wrapped"
        ]
    else:
        raise ValueError("Invalid dimension")

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
            "processor":      "CPU",
            "cpp_header":     "cpu_tf_hydro_Cpp2C.h",
            "cpp_source":     "cpu_tf_hydro_Cpp2C.cxx",
            "c2f_source":     "cpu_tf_hydro_C2F.F90",
            "fortran_source": "cpu_tf_hydro_mod.F90"
        },
        "data_item": {
            "type":           "TileWrapper",
            "byte_alignment": -1,
            "header":         "Tile_cpu_tf_hydro.h",
            "source":         "Tile_cpu_tf_hydro.cxx"
        }
    }

    # ----- ADJUST SUBROUTINE GROUP SPECIFICATION TO SPECIFIC PROBLEM
    group_json = group_spec_path.joinpath("Hydro_op1.json")
    if not group_json.is_file():
        msg = f"{group_json} does not exist or is not a file"
        raise ValueError(msg)
    with open(group_json, "r") as fptr:
        group_spec = json.load(fptr)

    # Scratch extents change with dimension
    sz_x = block_size[0] + 2
    sz_y = block_size[1] + 2 if dimension >= 2 else 1
    sz_z = block_size[2] + 2 if dimension == 3 else 1
    extents = f"({sz_x}, {sz_y}, {sz_z})"
    group_spec["scratch"]["_auxC"]["extents"] = extents

    sz_x = 1
    sz_y = 1 if dimension >= 2 else 0
    sz_z = 1 if dimension == 3 else 0
    lbound = f"(tile_lo) - ({sz_x}, {sz_y}, {sz_z})"
    group_spec["scratch"]["_auxC"]["lbound"] = lbound

    for i, each in enumerate(["_flX", "_flY", "_flZ"]):
        fl_size = block_size.copy()
        fl_size[i] += 1

        sz_x = fl_size[0] if i < dimension else 1
        sz_y = fl_size[1] if i < dimension else 1
        sz_z = fl_size[2] if i < dimension else 1
        n_flux = 5 if i < dimension else 1

        extents = f"({sz_x}, {sz_y}, {sz_z}, {n_flux})"
        group_spec["scratch"][each]["extents"] = extents

        lbound = "(tile_lo, 1)" if i < dimension else "(1, 1, 1, 1)"
        group_spec["scratch"][each]["lbound"] = lbound

    # Dump final operation specification
    #
    # This is not for this function to do its job.  However, it is a necessary
    # side effect for TestTaskFunctionAssembler
    filename = f"Hydro_op1_{dimension}D.json"
    group_json = destination.joinpath(filename)
    if (not overwrite) and group_json.exists():
        raise ValueError(f"{group_json} already exists")
    with open(group_json, "w") as fptr:
        json.dump(group_spec, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    group_spec = SubroutineGroup(group_spec, logger)

    # ----- DUMP PARTIAL TF SPECIFICATION
    filename = f"cpu_tf_hydro_partial_{dimension}D.json"
    partial_tf_spec_json = destination.joinpath(filename)
    if (not overwrite) and partial_tf_spec_json.exists():
        msg = f"{partial_tf_spec_json} already exists"
        raise ValueError(msg)
    with open(partial_tf_spec_json, "w") as fptr:
        json.dump(PARTIAL_TF_SPEC, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    # ----- GENERATE TASK FUNCTION SPECIFICATION JSON
    full_tf_spec = destination.joinpath(f"cpu_tf_hydro_{dimension}D.json")
    assembler = TaskFunctionAssembler(
                    "cpu_tf_hydro", TF_CALL_GRAPH,
                    [group_spec], GRID_SPEC, logger
                )
    assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

    return full_tf_spec
