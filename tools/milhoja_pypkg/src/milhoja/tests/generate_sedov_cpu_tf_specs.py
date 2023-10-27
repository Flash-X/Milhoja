import json

from milhoja import TaskFunctionAssembler


def generate_sedov_cpu_tf_specs(dimension, block_size,
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

        # ----- ADJUST OPERATION SPECIFICATION TO SPECIFIC PROBLEM
        op_spec_json = op_spec_path.joinpath(f"{unit}_{operation}.json")
        if not op_spec_json.is_file():
            msg = f"{op_spec_json} does not exist or is not a file"
            raise ValueError(msg)
        with open(op_spec_json, "r") as fptr:
            op_spec = json.load(fptr)

        # Add in Grid Spec
        assert "grid" not in op_spec
        op_spec["grid"] = GRID_SPEC

        # Some argument lists change with dimension
        sub = "hy::computeFluxesHll"
        if sub in op_spec["operation"]:
            arg_list = op_spec["operation"][sub]["argument_list"]
            if dimension == 1:
                arg_list = [e for e in arg_list if e not in ["flY", "flZ"]]
                op_spec["operation"][sub]["argument_list"] = arg_list
                del op_spec["operation"][sub]["argument_specifications"]["flY"]
                del op_spec["operation"][sub]["argument_specifications"]["flZ"]
            elif dimension == 2:
                arg_list = [e for e in arg_list if e not in ["flZ"]]
                op_spec["operation"][sub]["argument_list"] = arg_list
                del op_spec["operation"][sub]["argument_specifications"]["flZ"]

        # Scratch extents change with dimension
        if ("scratch" in op_spec["operation"]) and \
                ("_auxC" in op_spec["operation"]["scratch"]):
            sz_x = block_size[0] + 2
            sz_y = block_size[1] + 2 if dimension >= 2 else 1
            sz_z = block_size[2] + 2 if dimension == 3 else 1
            extents = f"({sz_x}, {sz_y}, {sz_z})"
            op_spec["operation"]["scratch"]["_auxC"]["extents"] = extents

            sz_x = 1
            sz_y = 1 if dimension >= 2 else 0
            sz_z = 1 if dimension == 3 else 0
            lbound = f"(tile_lo) - ({sz_x}, {sz_y}, {sz_z})"
            op_spec["operation"]["scratch"]["_auxC"]["lbound"] = lbound

        # Dump final operation specification for immediate use
        filename = f"{unit}_{operation}_{dimension}D.json"
        op_spec_json = destination.joinpath(filename)
        if (not overwrite) and op_spec_json.exists():
            raise ValueError(f"{op_spec_json} already exists")
        with open(op_spec_json, "w") as fptr:
            json.dump(op_spec, fptr,
                      ensure_ascii=True, allow_nan=False, indent=True)

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
        assembler = TaskFunctionAssembler.from_milhoja_json(
                        name, tf_call_graph, op_spec_json,
                        logger
                    )
        assembler.to_milhoja_json(full_tf_spec, partial_tf_spec_json, overwrite)

        tf_spec_jsons.append(full_tf_spec)

    return tf_spec_jsons
