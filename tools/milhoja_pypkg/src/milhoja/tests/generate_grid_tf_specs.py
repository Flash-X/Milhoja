import json

from milhoja import TaskFunctionAssembler


def generate_grid_tf_specs(dimension, block_size,
                           op_spec_path, destination, overwrite,
                           logger):
    """
    Use the partial operation specification located in the given path to fill
    in problem-specific operation specifications and construct the full
    specification JSON file for the task function needed to solve the given
    Grid/general test problem.

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
    TF_CALL_GRAPH = ["StaticPhysicsRoutines::setInitialConditions"]

    GRID_SPEC = {
        "dimension": dimension,
        "nxb": block_size[0],
        "nyb": block_size[1],
        "nzb": block_size[2],
        "nguardcells": 1
    }

    PARTIAL_TF_SPEC = {
        "task_function": {
            "language":       "C++",
            "processor":      "CPU",
            "cpp_header":     "cpu_tf_ic.h",
            "cpp_source":     "cpu_tf_ic.cpp",
            "c2f_source":     "",
            "fortran_source": ""
        },
        "data_item": {
            "type":           "TileWrapper",
            "byte_alignment": -1,
            "header":         "Tile_cpu_tf_ic.h",
            "source":         "Tile_cpu_tf_ic.cpp"
        }
    }

    # ----- ADJUST OPERATION SPECIFICATION TO SPECIFIC PROBLEM
    op_spec_json = op_spec_path.joinpath("Simulation_op1.json")
    if not op_spec_json.is_file():
        msg = f"{op_spec_json} does not exist or is not a file"
        raise ValueError(msg)
    with open(op_spec_json, "r") as fptr:
        op_spec = json.load(fptr)

    # Add in Grid Spec
    assert "grid" not in op_spec
    op_spec["grid"] = GRID_SPEC

    # Dump final operation specification for immediate use
    filename = f"Simulation_op1_{dimension}D.json"
    op_spec_json = destination.joinpath(filename)
    if (not overwrite) and op_spec_json.exists():
        raise ValueError(f"{op_spec_json} already exists")
    with open(op_spec_json, "w") as fptr:
        json.dump(op_spec, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    # ----- DUMP PARTIAL TF SPECIFICATION
    filename = f"cpu_tf_ic_partial_{dimension}D.json"
    partial_tf_spec_json = destination.joinpath(filename)
    if (not overwrite) and partial_tf_spec_json.exists():
        msg = f"{partial_tf_spec_json} already exists"
        raise ValueError(msg)
    with open(partial_tf_spec_json, "w") as fptr:
        json.dump(PARTIAL_TF_SPEC, fptr,
                  ensure_ascii=True, allow_nan=False, indent=True)

    # ----- GENERATE TASK FUNCTION SPECIFICATION JSON
    full_tf_spec = destination.joinpath(f"cpu_tf_ic_{dimension}D.json")
    assembler = TaskFunctionAssembler.from_milhoja_json(
                    "cpu_tf_ic", TF_CALL_GRAPH, op_spec_json,
                    logger
                )
    assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

    return full_tf_spec
