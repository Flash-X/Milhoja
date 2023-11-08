import json

from milhoja import (
    SubroutineGroup,
    TaskFunctionAssembler
)


def generate_runtime_cpu_tf_specs(group_spec_path, destination, overwrite,
                                  logger):
    """
    Construct the full specification JSON file for each task function needed to
    solve the given Runtime/CPU test problem using the problem's subroutine
    group specification files.

    :param group_spec_path: Path to location of Runtime/CPU problem's
        subroutine group specification JSON files
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing JSON files in destination will be overwritten
        if True
    :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
    """
    # ----- HARDCODED
    GRID_SPEC = {
        "dimension": 2,
        "nxb": 8,
        "nyb": 16,
        "nzb": 1,
        "nguardcells": 1
    }

    MILHOJA_INVOCATIONS = [
        ("cpu_tf_ic", "Simulation", "op1"),
        ("cpu_tf_dens", "Math", "op1"),
        ("cpu_tf_ener", "Math", "op1"),
        ("cpu_tf_fused", "Math", "op1"),
        ("cpu_tf_analysis", "Analysis", "op1")
    ]

    CALL_GRAPHS = {
        "cpu_tf_ic": ["StaticPhysicsRoutines::setInitialConditions"],
        "cpu_tf_dens": ["StaticPhysicsRoutines::computeLaplacianDensity"],
        "cpu_tf_ener": ["StaticPhysicsRoutines::computeLaplacianEnergy"],
        "cpu_tf_fused": ["StaticPhysicsRoutines::computeLaplacianFusedKernels"],
        "cpu_tf_analysis": ["Analysis::computeErrors"]
    }

    # ----- BOILERPLATE SPECIFICATIONS
    partial_tf_spec = {
        "task_function": {
            "language":       "C++",
            "processor":      "CPU",
            "c2f_source":     "",
            "fortran_source": ""
        },
        "data_item": {
            "type":           "TileWrapper",
            "byte_alignment": -1
        }
    }

    tf_spec_jsons = []
    for name, unit, operation in MILHOJA_INVOCATIONS:
        tf_call_graph = CALL_GRAPHS[name]
        group_json = group_spec_path.joinpath(f"{unit}_{operation}.json")
        group_spec = SubroutineGroup.from_milhoja_json(group_json, logger)

        # ----- UPDATE PARTIAL TF SPECIFICATION FOR SPECIFIC PROBLEM
        partial_tf_spec["task_function"]["cpp_header"] = f"{name}.h"
        partial_tf_spec["task_function"]["cpp_source"] = f"{name}.cpp"
        partial_tf_spec["data_item"]["header"] = f"Tile_{name}.h"
        partial_tf_spec["data_item"]["source"] = f"Tile_{name}.cpp"

        filename = f"{name}_partial.json"
        partial_tf_spec_json = destination.joinpath(filename)
        if (not overwrite) and partial_tf_spec_json.exists():
            raise ValueError(f"{partial_tf_spec} already exists")
        with open(partial_tf_spec_json, "w") as fptr:
            json.dump(partial_tf_spec, fptr,
                      ensure_ascii=True, allow_nan=False, indent=True)

        # ----- GENERATE TASK FUNCTION SPECIFICATION JSON
        full_tf_spec = destination.joinpath(f"{name}.json")
        assembler = TaskFunctionAssembler(
                        name, tf_call_graph, [group_spec], GRID_SPEC,
                        logger
                    )
        assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

        tf_spec_jsons.append(full_tf_spec)

    return tf_spec_jsons
