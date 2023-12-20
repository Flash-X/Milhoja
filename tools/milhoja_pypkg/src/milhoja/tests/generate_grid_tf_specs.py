import json

from milhoja import (
    SubroutineGroup,
    TaskFunctionAssembler
)


def generate_grid_tf_specs(dimension, block_size,
                           group_spec_path, destination, overwrite,
                           logger):
    """
    Construct the full specification JSON file for the task function needed to
    solve the given Grid/general test problem using the given information and
    the problem's subroutine group specification JSON file.

    :param dimension: Dimension of problem's domain
    :param block_size: (nxb, nyb, nzb) where n[xyz]b is N cells in each block
        along associated dimension of problem's domain
    :param group_spec_path: Path to location of Grid/general problem's
        subroutine group specification JSON files
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing JSON files in destination will be overwritten
        if True
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
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
            "source":         "Tile_cpu_tf_ic.cpp",
            "module":         ""
        }
    }

    GROUP_JSON = group_spec_path.joinpath("Simulation_op1.json")

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
    group_spec = SubroutineGroup.from_milhoja_json(GROUP_JSON, logger)

    full_tf_spec = destination.joinpath(f"cpu_tf_ic_{dimension}D.json")
    assembler = TaskFunctionAssembler(
                    "cpu_tf_ic", TF_CALL_GRAPH, [group_spec], GRID_SPEC,
                    logger
                )
    assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

    return full_tf_spec
