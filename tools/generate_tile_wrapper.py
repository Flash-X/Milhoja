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
    DESCRIPTION = "Generate the .h/.cpp Milhoja tile wrapper code associated\n" \
                  "with the given JSON task function specification file.\n"
    JSON_HELP = "JSON-format file that fully specifies a task function\n"
    HEADER_HELP = "Filename of the header file to generate\n"
    SOURCE_HELP = "Filename of the source file to generate\n"
    VERBOSE_HELP = f"Verbosity level of logging.  Valid values are {milhoja.CodeGenerationLogger.LOG_LEVELS}."

    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("json",   nargs=1, help=JSON_HELP)
    parser.add_argument("header", nargs=1, help=HEADER_HELP)
    parser.add_argument("source", nargs=1, help=SOURCE_HELP)
    parser.add_argument("--verbose", "-v", type=int, help=VERBOSE_HELP, \
                        default=milhoja.CodeGenerationLogger.BASIC_LOG_LEVEL)

    #####----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    json_filename   = Path(args.json[0]).resolve()
    header_filename = Path(args.header[0]).resolve()
    source_filename = Path(args.source[0]).resolve()
    verbosity_level = args.verbose

    #####----- LOGGING
    try:
        logger = milhoja.CodeGenerationLogger( \
                                "TileWrapper Generator", \
                                verbosity_level \
                            )
    except Exception as error:
        # Assume that the logger printed the error message itself already
        exit(1)

    def print_and_abort(error_msg):
        logger.error(error_msg)
        exit(1)

    #####----- ERROR CHECKING
    if not json_filename.is_file():
        print_and_abort(f"{json_filename} does not exist or is not a file")
    if header_filename.exists():
        print_and_abort(f"{header_filename} already exists")
    if source_filename.exists():
        print_and_abort(f"{source_filename} already exists")

    #####----- GET TO GENERATIN'
    try:
        generator = milhoja.TileWrapperGenerator.from_json( \
                        json_filename, \
                        header_filename, \
                        source_filename, \
                        logger, \
                        indent=INDENT \
                    )
        generator.generate_header_code()
        generator.generate_source_code()
    except Exception as error:
        error_msg = str(error)
        if logger.level >= milhoja.CodeGenerationLogger.BASIC_DEBUG_LEVEL:
            error_msg += f"\n{traceback.format_exc()}"
        print_and_abort(error_msg)

    if not header_filename.is_file():
        print_and_abort(f"Header file not generated ({header_filename})")
    if not source_filename.is_file():
        print_and_abort(f"Source file not generated ({source_filename})")

if __name__ == "__main__":
    main()

