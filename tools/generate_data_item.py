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

    INDENT = 4
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    DESCRIPTION = "Generate Milhoja data item code needed to support the\n" \
                  "task function specified in the given file"
    FILENAME_HELP = "Task function specification file"
    FORMAT_HELP = "Task function specification format"
    DESTINATION_HELP = "Pre-existing folder to write files to"
    OVERWRITE_HELP = "Original files overwritten if given"
    VERBOSE_HELP = f"Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=formatter)
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument("format", nargs=1,
                        type=str, choices=milhoja.TASK_FUNCTION_FORMATS,
                        help=FORMAT_HELP)
    parser.add_argument("destination", nargs=1, help=DESTINATION_HELP)
    parser.add_argument("--overwrite",
                        action='store_true', required=False, \
                        help=OVERWRITE_HELP)
    parser.add_argument("--verbose", "-v",
                        type=int, choices=milhoja.LOG_LEVELS,
                        help=VERBOSE_HELP,
                        default=DEFAULT_LOG_LEVEL)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    filename = Path(args.file[0]).resolve()
    fmt = args.format[0]
    destination = Path(args.destination[0]).resolve()
    overwrite = args.overwrite
    verbosity_level = args.verbose

    # ----- ABORT WITH MESSAGE & COMMUNICATE FAILURE
    def print_and_abort(error_msg):
        FAILURE = '\033[0;91;1m'  # Bright Red/bold
        NC = '\033[0m'            # No Color/Not bold
        print()
        print(f"{FAILURE}ERROR - {error_msg}{NC}")
        print()
        exit(FAILURE)

    # ----- GET TO GENERATIN'
    try:
        if fmt.lower() == milhoja.MILHOJA_JSON_FORMAT.lower():
            tf_spec = milhoja.TaskFunction.from_milhoja_json(filename, fmt)
        else:
            print_and_abort(f"Unsupported task function format ({fmt})")

        milhoja.generate_data_item(
            tf_spec,
            destination, overwrite,
            verbosity_level, INDENT
        )
    except Exception as error:
        error_msg = str(error)
        if verbosity_level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    return SUCCESS

if __name__ == "__main__":
    exit(main())
