#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import argparse
import traceback

from pathlib import Path

import milhoja


def main():
    # ----- HARDCODED VALUES
    # Exit codes so that this can be used in CI build server
    FAILURE = 1
    SUCCESS = 0

    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "DO NOT USE.  JUST A PLACEHOLDER FOR POTENTIAL FUTURE."
    FILENAME_HELP = "Task function specification file to convert"
    FROM_HELP = "Format of task function specification"
    TO_HELP = "Desired task function specification"
    VERBOSE_HELP = "Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
                description=DESCRIPTION,
                formatter_class=formatter
             )
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument(
        "from_format", nargs=1,
        type=str, choices=milhoja.TASK_FUNCTION_FORMATS,
        help=FROM_HELP
    )
    parser.add_argument(
        "to_format", nargs=1,
        type=str, choices=milhoja.TASK_FUNCTION_FORMATS,
        help=TO_HELP
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=milhoja.LOG_LEVELS,
        help=VERBOSE_HELP,
        default=DEFAULT_LOG_LEVEL
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    filename = Path(args.file[0]).resolve()
    from_format = args.from_format
    to_format = args.to_format
    verbosity_level = args.verbose

    def print_and_abort(error_msg):
        FAILURE_COLOR = '\033[0;91;1m'  # Bright Red/bold
        NC = '\033[0m'                  # No Color/Not bold
        print()
        print(f"{FAILURE_COLOR}ERROR - {error_msg}{NC}")
        print()
        exit(FAILURE)

    # ----- GET TO CONVERTIN'
    if from_format.lower() == to_format.lower():
        print_and_abort("To and from formats are identical")
    else:
        print_and_abort("This has never been tested")

    try:
        if from_format.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            tf_spec = milhoja.TaskFunction.from_milhoja_json(filename)
        else:
            # This should never happen because argparse should error first
            error_msg = f"Unknown from specification format {from_format}"
            print_and_abort(error_msg)

        if to_format.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            # TODO: Determine new filename from original and format?
            filename_dst = "delete_me.json"
            tf_spec.to_milhoja_json(filename_dst)
        else:
            # This should never happen because argparse should error first
            error_msg = f"Unknown to specification format {to_format}"
            print_and_abort(error_msg)
    except Exception as error:
        error_msg = str(error)
        if verbosity_level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    return SUCCESS


if __name__ == "__main__":
    exit(main())
