#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""
import json
import argparse
import traceback

from pathlib import Path

import milhoja


def main():
    # ----- HARDCODED VALUES
    DESTINATION = Path(__file__).resolve().parent

    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    INDENT = 4
    LOG_TAG = "Flash-X Orchestration"
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    TF_NAME = "cpu_tf_hydro"
    HYDRO_OP1_JSON = DESTINATION.joinpath("__Hydro_op_spec.json")
    TF_JSON = DESTINATION.joinpath(f"{TF_NAME}.json")
    PARTIAL_TF_JSON = DESTINATION.joinpath(f"{TF_NAME}_partial.json")
    GRID_JSON = DESTINATION.joinpath("grid.json")

    PARTIAL_TF_SPEC = {
        "task_function": {
            "language":       "Fortran",
            "processor":      "CPU",
            "cpp_header":     f"{TF_NAME}_Cpp2C.h",
            "cpp_source":     f"{TF_NAME}_Cpp2C.cxx",
            "c2f_source":     f"{TF_NAME}_C2F.F90",
            "fortran_source": f"{TF_NAME}_mod.F90",
            "computation_offloading": "",
            "variable_index_base": 1
        },
        "data_item": {
            "type":           "TileWrapper",
            "byte_alignment": -1,
            "header":         f"Tile_{TF_NAME}.h",
            "source":         f"Tile_{TF_NAME}.cxx",
            "module":         f"Tile_{TF_NAME}_C2F_mod.F90"
        }
    }

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Generate Sedov code for use with Orchestration unit"
    DIM_HELP = "Dimension of problem"
    NXB_HELP = "N cells in each block along x-axis"
    NYB_HELP = "N cells in each block along y-axis"
    NZB_HELP = "N cells in each block along z-axis"
    LIBRARY_HELP = "Path to Milhoja library installation that will use code"
    OVERWRITE_HELP = "Original files overwritten if given"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=formatter)
    parser.add_argument(
        "dimension", nargs=1, type=int, choices=[1, 2, 3],
        help=DIM_HELP
    )
    parser.add_argument("nxb", nargs=1, type=int, help=NXB_HELP)
    parser.add_argument("nyb", nargs=1, type=int, help=NYB_HELP)
    parser.add_argument("nzb", nargs=1, type=int, help=NZB_HELP)
    parser.add_argument("milhoja_path", nargs=1, help=LIBRARY_HELP)
    parser.add_argument(
        "--overwrite", "-o", action='store_true', required=False,
        help=OVERWRITE_HELP
    )
    parser.add_argument(
        "--verbose", "-v", type=int,
        choices=milhoja.LOG_LEVELS, default=DEFAULT_LOG_LEVEL,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    dimension = args.dimension[0]
    nxb = args.nxb[0]
    nyb = args.nyb[0]
    nzb = args.nzb[0]
    milhoja_path = Path(args.milhoja_path[0]).resolve()
    overwrite = args.overwrite

    GRID_SPEC = {
        "dimension": dimension,
        "nxb": nxb,
        "nyb": nyb,
        "nzb": nzb,
        "nguardcells": 1
    }

    TF_CALL_GRAPH = [
        "Hydro_prepBlock",
        "Hydro_advance"
    ]
    logger = milhoja.BasicLogger(args.verbose)

    # ----- CONSTRUCT HELPER FUNCTIONS
    def log(msg):
        logger.log(LOG_TAG, msg, milhoja.LOG_LEVEL_BASIC)

    def warn(msg):
        logger.warn(LOG_TAG, msg)

    def log_and_abort(error_msg):
        logger.error(LOG_TAG, error_msg)
        exit(FAILURE)

    # ----- ADJUST HYDRO/OPERATION 1 SPECIFICATION TO SPECIFIC PROBLEM
    with open(HYDRO_OP1_JSON, "r") as fptr:
        hydro_op1_spec = json.load(fptr)

    # ----- WRITE ALL FILES
    contents_all = [
        ("grid specification", GRID_SPEC, GRID_JSON),
        ("partial TF specification", PARTIAL_TF_SPEC, PARTIAL_TF_JSON),
        ("Hydro/Operation 1 specification", hydro_op1_spec, HYDRO_OP1_JSON)
    ]
    for desc, content, filename in contents_all:
        if filename.exists():
            if overwrite:
                warn(f"{filename} overwritten")
            else:
                log_and_abort(f"{filename} already exists")
        log(f"Write {desc} to {filename}")
        with open(filename, "w") as fptr:
            json.dump(content, fptr,
                      ensure_ascii=True, allow_nan=False, indent=True)

    # ----- GENERATE HYDRO ADVANCE ACTION CODE FOR ORCHESTRATION SYSTEM
    log("Generate Hydro/Operation 1 code for Orchestration unit")
    try:
        assembler = milhoja.TaskFunctionAssembler.from_milhoja_json(
                        TF_NAME, TF_CALL_GRAPH, [HYDRO_OP1_JSON],
                        GRID_JSON, logger
                    )
        assembler.to_milhoja_json(TF_JSON, PARTIAL_TF_JSON, overwrite)

        tf_spec = milhoja.TaskFunction.from_milhoja_json(TF_JSON)

        milhoja.generate_data_item(
           tf_spec, DESTINATION, overwrite, milhoja_path, INDENT, logger
        )
        # TODO: Generate task function once in package
        milhoja.generate_task_function(
            tf_spec, DESTINATION, overwrite, INDENT, logger
        )
    except Exception as error:
        error_msg = str(error)
        if logger.level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        log_and_abort(error_msg)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
