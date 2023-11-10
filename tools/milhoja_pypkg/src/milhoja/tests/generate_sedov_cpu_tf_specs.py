import json

from milhoja import (
    SubroutineGroup,
    TaskFunctionAssembler
)


def generate_sedov_cpu_tf_specs(dimension, block_size,
                                group_spec_path, destination, overwrite,
                                logger):
    """
    Use the partial subroutine group specifications located in the given path
    to fill in problem-specific subroutine group specifications and construct
    the full specification JSON file for each task function needed to solve the
    given Sedov test problem.

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
    MILHOJA_INVOCATIONS = [
        ("cpu_tf_ic", "Simulation", "op1"),
        ("cpu_tf_hydro", "Hydro", "op1"),
        ("cpu_tf_IQ", "Io", "op1")
    ]

    CALL_GRAPHS = {
        "cpu_tf_ic": ["sim::setInitialConditions", "Eos::idealGammaDensIe"],
        "cpu_tf_hydro": [
            "hy::computeFluxesHll",
            "hy::updateSolutionHll",
            "Eos::idealGammaDensIe"
        ],
        "cpu_tf_IQ": ["Io::instance().computeIntegralQuantitiesByBlock"]
    }

    GRID_SPEC = {
        "dimension": dimension,
        "nxb": block_size[0],
        "nyb": block_size[1],
        "nzb": block_size[2],
        "nguardcells": 1
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

        # ----- ADJUST SUBROUTINE GROUP SPECIFICATION TO SPECIFIC PROBLEM
        group_json = group_spec_path.joinpath(f"{unit}_{operation}.json")
        if not group_json.is_file():
            msg = f"{group_json} does not exist or is not a file"
            raise ValueError(msg)
        with open(group_json, "r") as fptr:
            group_spec = json.load(fptr)

        # Some argument lists change with dimension
        for sub in ["hy::computeFluxesHll", "hy::updateSolutionHll"]:
            if sub in group_spec:
                arg_spec_key = "argument_specifications"
                arg_list = group_spec[sub]["argument_list"]
                if dimension == 1:
                    arg_list = [e for e in arg_list if e not in ["flY", "flZ"]]
                    group_spec[sub]["argument_list"] = arg_list
                    del group_spec[sub][arg_spec_key]["flY"]
                    del group_spec[sub][arg_spec_key]["flZ"]
                elif dimension == 2:
                    arg_list = [e for e in arg_list if e not in ["flZ"]]
                    group_spec[sub]["argument_list"] = arg_list
                    del group_spec[sub][arg_spec_key]["flZ"]

        # Scratch extents change with dimension
        if ("scratch" in group_spec) and ("_auxC" in group_spec["scratch"]):
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

        group_spec = SubroutineGroup(group_spec, logger)

        # ----- UPDATE PARTIAL TF SPECIFICATION FOR SPECIFIC PROBLEM
        partial_tf_spec["task_function"]["cpp_header"] = f"{name}.h"
        partial_tf_spec["task_function"]["cpp_source"] = f"{name}.cpp"
        partial_tf_spec["data_item"]["header"] = f"Tile_{name}.h"
        partial_tf_spec["data_item"]["source"] = f"Tile_{name}.cpp"

        filename = f"{name}_partial_{dimension}D.json"
        partial_tf_spec_json = destination.joinpath(filename)
        if (not overwrite) and partial_tf_spec_json.exists():
            msg = f"{partial_tf_spec} already exists"
            raise ValueError(msg)
        with open(partial_tf_spec_json, "w") as fptr:
            json.dump(partial_tf_spec, fptr,
                      ensure_ascii=True, allow_nan=False, indent=True)

        # ----- GENERATE TASK FUNCTION SPECIFICATION JSON
        full_tf_spec = destination.joinpath(f"{name}_{dimension}D.json")
        assembler = TaskFunctionAssembler(
                        name, tf_call_graph, [group_spec], GRID_SPEC,
                        logger
                    )
        assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

        tf_spec_jsons.append(full_tf_spec)

    return tf_spec_jsons
