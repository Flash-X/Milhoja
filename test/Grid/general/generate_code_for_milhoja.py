#!/usr/bin/env python

import argparse

from pathlib import Path

import milhoja.tests

def main():
    # ----- HARDCODED VALUES
    CLONE_PATH = Path(__file__).resolve().parents[3]

    # Location of Milhoja-JSON files
    CG_PATH = CLONE_PATH.joinpath("test", "Grid", "general", "code_generation")

    # Milhoja-JSON specifications of all task functions in test
    TF_PARTIAL_NAMES_ALL = ["cpu_tf_ic_{}D.json"]

    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    INDENT = 4
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Milhoja's test build system calls this to generate code"
    DESTINATION_HELP = "Pre-existing folder to write files to"
    DIM_HELP = "Dimension of test problem"
    MAKEFILE_HELP = "Filename with path of Makefile to generate"
    OVERWRITE_HELP = "Original files overwritten if given"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
                description=DESCRIPTION, formatter_class=formatter
             )
    parser.add_argument("destination", nargs=1, help=DESTINATION_HELP)
    parser.add_argument("makefile", nargs=1, help=MAKEFILE_HELP)
    parser.add_argument(
        "dimension", nargs=1, type=int, choices=[1, 2, 3],
        help=DIM_HELP
    )
    parser.add_argument(
        "--overwrite", action='store_true', required=False,
        help=OVERWRITE_HELP
    )
    parser.add_argument(
        "--verbose", "-v", type=int,
        choices=milhoja.LOG_LEVELS, default=DEFAULT_LOG_LEVEL,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    destination = Path(args.destination[0]).resolve()
    makefile = Path(args.makefile[0]).resolve()
    dimension = args.dimension[0]
    overwrite = args.overwrite
    verbosity_level = args.verbose

    tf_names_all = [each.format(dimension) for each in TF_PARTIAL_NAMES_ALL]

    # ----- ABORT WITH MESSAGE & COMMUNICATE FAILURE
    def print_and_abort(error_msg):
        FAILURE = '\033[0;91;1m'  # Bright Red/bold
        NC = '\033[0m'            # No Color/Not bold
        print()
        print(f"{FAILURE}ERROR - {error_msg}{NC}")
        print()
        exit(FAILURE)

    try:
        # ----- LOAD ALL TASK FUNCTIONS SPECIFICATIONS
        tf_specs_all = []
        for tf_name in tf_names_all:
            filename = CG_PATH.joinpath(tf_name)
            tf_spec = milhoja.TaskFunction.from_milhoja_json(filename)
            tf_specs_all.append(tf_spec)

        # ----- NOW IS GOOD FOR GENERATING CODE
        milhoja.tests.generate_code(
            tf_specs_all, destination,
            overwrite, verbosity_level, INDENT,
            makefile
        )
    except Exception as error:
        error_msg = str(error)
        if verbosity_level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    return SUCCESS

if __name__ == "__main__":
    exit(main())
