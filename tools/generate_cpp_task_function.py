#!/usr/bin/env python

"""
Run the script with -h to obtain more information regarding the script.
"""

import argparse
import traceback

from pathlib import Path

import milhoja

def main():
    #####----- HARDCODED VALUES
    INDENT = 4

    #####----- PROGRAM USAGE INFO
    DESCRIPTION = "Generate the .h/.cpp Milhoja task function code associated\n" \
                  "with the given JSON task function specification file.\n"
    JSON_HELP = "JSON-format file that fully specifies a CPU/C++ task function\n"
    HEADER_HELP = "Filename of the header file to generate\n"
    SOURCE_HELP = "Filename of the source file to generate\n"
    VERBOSE_HELP = f"Verbosity level of logging"

    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("json",   nargs=1, help=JSON_HELP)
    parser.add_argument("header", nargs=1, help=HEADER_HELP)
    parser.add_argument("source", nargs=1, help=SOURCE_HELP)
    parser.add_argument("--verbose", "-v",
                        type=int, choices=milhoja.LOG_LEVELS,
                        help=VERBOSE_HELP,
                        default=milhoja.LOG_LEVEL_BASIC)

    #####----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    json_filename   = Path(args.json[0]).resolve()
    header_filename = Path(args.header[0]).resolve()
    source_filename = Path(args.source[0]).resolve()
    verbosity_level = args.verbose

    def print_and_abort(error_msg):
        FAILURE = '\033[0;91;1m'  # Bright Red/bold
        NC = '\033[0m'            # No Color/Not bold
        print()
        print(f"{FAILURE}ERROR - {error_msg}{NC}")
        print()
        exit(1)

    #####----- GET TO GENERATIN'
    # TODO: We should be able to determine which generator to use based on the
    # extension of the source file or the passing of a header file.
    try:
        tf_spec = milhoja.TaskFunction.from_json(json_filename)

        generator = milhoja.CppTaskFunctionGenerator.from_json(
                        json_filename,
                        header_filename,
                        source_filename,
                        verbosity_level,
                        indent=INDENT
                    )
        generator.generate_header_code()
        generator.generate_source_code()
    except Exception as error:
        error_msg = str(error)
        if verbosity_level >= milhoja.LOG_LEVEL_BASIC_DEBUG:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    if not header_filename.is_file():
        print_and_abort(f"Header file not generated ({header_filename})")
    if not source_filename.is_file():
        print_and_abort(f"Source file not generated ({source_filename})")

if __name__ == "__main__":
    main()

