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
    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    LOG_TAG = "Milhoja Tools"
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC_DEBUG

    # ANSI Terminal Colors
    # Rather than use green/red, I have been told that blue/red is better
    # for people with color blindness
    SUCCESS_COLOR = '\033[0;94;1m'  # Bright Blue/bold
    NO_COLOR = '\033[0m'            # No Color/Not bold

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Check the given operation specification file for errors"
    FILENAME_HELP = "Operation specification file to check"
    FROM_HELP = "Format of operation specification file"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=formatter)
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument(
        "from_format", nargs=1, type=str.lower,
        choices=[e.lower() for e in milhoja.TASK_FUNCTION_FORMATS],
        help=FROM_HELP
    )
    parser.add_argument("--verbose", "-v", type=int,
                        choices=milhoja.LOG_LEVELS, help=VERBOSE_HELP,
                        default=DEFAULT_LOG_LEVEL)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    op_spec = Path(args.file[0]).resolve()
    from_format = args.from_format[0]
    logger = milhoja.BasicLogger(args.verbose)

    def log_and_abort(error_msg):
        print()
        logger.error(LOG_TAG, error_msg)
        print()
        exit(FAILURE)

    if not op_spec.is_file():
        log_and_abort(f"{op_spec} does not exist or is not a file")

    # ----- CHECK SPECIFICATION
    try:
        if from_format.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            # -- Construct Fake Call Graph
            with open(op_spec, "r") as fptr:
                spec = json.load(fptr)
            if "operation" not in spec:
                log_and_abort(f"{op_spec} missing 'operation' key")
            op_spec = spec["operation"]

            # Find all subroutines
            ignore = {"name", "variable_index_base"}
            for key in ignore:
                if key not in op_spec:
                    log_and_abort(f"'{key}' not specified in operation")
            if milhoja.EXTERNAL_ARGUMENT in op_spec:
                ignore = ignore.union([milhoja.EXTERNAL_ARGUMENT])
            if milhoja.SCRATCH_ARGUMENT in op_spec:
                ignore = ignore.union([milhoja.SCRATCH_ARGUMENT])

            subroutines_all = set(op_spec).difference(ignore)
            if len(subroutines_all) == 0:
                log_and_abort("No subroutines in specification")

            # Pick one subroutine to build graph
            call_graph = [sorted(subroutines_all)[0]]

            # -- Loading runs the full check
            name = LOG_TAG.replace(' ', '_')
            milhoja.TaskFunctionAssembler.from_milhoja_json(name, call_graph,
                                                            op_spec, logger)
        else:
            # This should never happen because argparse should error first
            error_msg = f"Unknown from specification format {from_format}"
            log_and_abort(error_msg)
    except Exception as error:
        error_msg = str(error)
        if logger.level >= (milhoja.LOG_LEVEL_BASIC_DEBUG + 1):
            # Don't show traceback by default
            error_msg += f"\n{traceback.format_exc()}"
        log_and_abort(error_msg)

    msg = f"{SUCCESS_COLOR}Specification Valid{NO_COLOR}"
    logger.log(LOG_TAG, "", milhoja.LOG_LEVEL_BASIC)
    logger.log(LOG_TAG, msg, milhoja.LOG_LEVEL_BASIC)
    logger.log(LOG_TAG, "", milhoja.LOG_LEVEL_BASIC)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
