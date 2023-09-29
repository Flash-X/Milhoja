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
    INDENT = 4
    DEFAULT_LOG_LEVEL = milhoja.LOG_LEVEL_BASIC

    # ----- PROGRAM USAGE INFO
    # TODO: Move output file names to specification file
    DESCRIPTION = "Generate Milhoja data item code needed to support the\n" \
                  "task function specified in the given file"
    FILENAME_HELP = "Task function specification file"
    FORMAT_HELP = "Task function specification format"
    HEADER_HELP = "Filename of the header file to generate"
    SOURCE_HELP = "Filename of the source file to generate"
    VERBOSE_HELP = f"Verbosity level of logging"

    # ----- SPECIFY COMMAND LINE USAGE
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=formatter)
    parser.add_argument("file", nargs=1, help=FILENAME_HELP)
    parser.add_argument("format", nargs=1,
                        type=str, choices=milhoja.TASK_FUNCTION_FORMATS,
                        help=FORMAT_HELP)
    parser.add_argument("header", nargs=1, help=HEADER_HELP)
    parser.add_argument("source", nargs=1, help=SOURCE_HELP)
    parser.add_argument("--verbose", "-v",
                        type=int, choices=milhoja.LOG_LEVELS,
                        help=VERBOSE_HELP,
                        default=DEFAULT_LOG_LEVEL)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    filename = Path(args.file[0]).resolve()
    fmt = args.format[0]
    header_filename = Path(args.header[0]).resolve()
    source_filename = Path(args.source[0]).resolve()
    verbosity_level = args.verbose

    # ----- ABORT WITH MESSAGE & COMMUNICATE FAILURE
    def print_and_abort(error_msg):
        FAILURE = '\033[0;91;1m'  # Bright Red/bold
        NC = '\033[0m'            # No Color/Not bold
        print()
        print(f"{FAILURE}ERROR - {error_msg}{NC}")
        print()
        exit(1)

    # ----- BE CONSERVATIVE - DON'T OVERWRITE
    # TODO: Once we generate Fortran files, see if these return a file or a
    # list of files.
    if header_filename.exists():
        print_and_abort(f"{header_filename} already exists")
    if source_filename.exists():
        print_and_abort(f"{source_filename} already exists")

    # ----- GET TO GENERATIN'
    try:
        milhoja.generate_data_item(
            filename, fmt,
            header_filename, source_filename,
            verbosity_level, INDENT
        )
    except Exception as error:
        error_msg = str(error)
        if verbosity_level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    return 0

if __name__ == "__main__":
    exit(main())
