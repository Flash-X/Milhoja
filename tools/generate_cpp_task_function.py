#!/usr/bin/env python3

"""
Run the script with -h to obtain more information regarding the script.
"""

import argparse

from pathlib import Path

import milhoja

def main():
    #####----- PROGRAM USAGE INFO
    __DESCRIPTION = "Generate the .h/.cpp Milhoja task function code associated\n" \
                    "with the given JSON task function specification file.\n"
    __JSON_HELP = "JSON-format file that fully specifies a CPU/C++ task function\n"
    __HEADER_HELP = "Filename of the header file to generate\n"
    __SOURCE_HELP = "Filename of the source file to generate\n"

    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=__DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('json',   nargs=1, help=__JSON_HELP)
    parser.add_argument('header', nargs=1, help=__HEADER_HELP)
    parser.add_argument('source', nargs=1, help=__SOURCE_HELP)

    def print_and_abort(error_msg):
        # ANSI terminal colors
        FAILURE  = '\033[0;91;1m' # Bright Red/bold
        NC       = '\033[0m'      # No Color/Not bold
        print()
        parser.print_help()
        print()
        print(f"{FAILURE}ERROR - {error_msg}{NC}")
        print()
        exit(1)

    #####----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    json_filename   = Path(args.json[0]).resolve()
    header_filename = Path(args.header[0]).resolve()
    source_filename = Path(args.source[0]).resolve()

    if not json_filename.is_file():
        print_and_abort(f"{json_filename} does not exist or is not a file")
    if header_filename.exists():
        print_and_abort(f"{header_filename} already exists")
    if source_filename.exists():
        print_and_abort(f"{source_filename} already exists")

    # TODO: We should be able to determine which generator to use based on the
    # extension of the source file or the passing of a header file.
    generator = milhoja.CppTaskFunctionGenerator( \
                    json_filename, \
                    header_filename, \
                    source_filename \
                )
    generator.generate_header_code()
    if not header_filename.is_file():
        print_and_abort(f"Header file not generated ({header_filename})")
    generator.generate_source_code()
    if not source_filename.is_file():
        print_and_abort(f"Source file not generated ({source_filename})")

if __name__ == '__main__':
    main()

