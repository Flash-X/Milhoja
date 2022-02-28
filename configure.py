#!/usr/bin/env python3

import sys
import argparse

from pathlib import Path

#####----- FIXED CONFIGURATION VALUES
# Always build with IEEE double precision reals.
_FLOATING_POINT_SYSTEM = 'double'

#####----- DEFAULT CONFIGURATION VALUES
# For no apparent reason, default is host-only execution.
_DEFAULT_GRID     = 'AMReX'
_DEFAULT_RUNTIME  = 'None'
_DEFAULT_OFFLOAD  = 'None'

#####----- PROGRAM USAGE INFO
_DESCRIPTION = \
    "This script can be used to configure the library build system so that\n" \
    "it can be used to build a particular flavor of library.  In particular,\n" \
    "it writes Makefile.configure to the same folder as this script.\n\n" \
    "To build the static library, run 'make all' from the root of the\n" \
    "repository.  After a successful build, `make install` will install the\n" \
    "library, headers, and Fortran mod files in the folder specified by the\n" \
    "user via the --prefix flag.\n"
_MAKEFILE_HELP = \
    '[mandatory] Full path to Makefile\n'
_RUNTIME_HELP = \
    'Specify the runtime backend to be used by the library.  Refer to the\n' \
    'write_library_header.py documentation for valid values.\n' \
   f'\tDefault: {_DEFAULT_RUNTIME}\n'
_GRID_HELP = \
    'Specify the block-structured adaptive mesh refinement library to use\n' \
    'as the Grid backend.  Refer to the write_library_header.py\n' \
    'documentation for valid values.\n' \
   f'\tDefault: {_DEFAULT_GRID}\n'
_OFFLOAD_HELP = \
    'Specify the tool that calling code will use to offload computation.\n' \
    'Refer to the write_library_header.py documentation for valid values.\n' \
   f'\tDefault: {_DEFAULT_OFFLOAD}\n'

#####----- ANSI TERMINAL COLORS
_ERROR = '\033[0;91;1m' # Bright Red/bold
_NC    = '\033[0m'      # No Color/Not bold

#####----- HARDCODED VARIABLES
# This file is located in the repository root directory.  We assemble the build
# relative to that directory.
_HOME_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    """
    Setup the library build system in accord with the given command line arguments.
    """
    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=_DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--makefile', '-M', type=str,                            help=_MAKEFILE_HELP)
    parser.add_argument('--dim',      '-d', type=int,                            help='[mandatory] Dimensionality of library.')
    parser.add_argument('--runtime',  '-r', type=str, default=_DEFAULT_RUNTIME,  help=_RUNTIME_HELP)
    parser.add_argument('--grid',     '-g', type=str, default=_DEFAULT_GRID,     help=_GRID_HELP)
    parser.add_argument('--offload',  '-o', type=str, default=_DEFAULT_OFFLOAD,  help=_OFFLOAD_HELP)
    parser.add_argument('--prefix',         type=str,                            help='[mandatory] Where to install library')
    parser.add_argument('--debug',          action="store_true", help='Set up in debug mode.')

    def print_and_exit(msg):
        print(file=sys.stderr)
        print(f'{_ERROR}SETUP ERROR: {msg}{_NC}', file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(1)

    #####----- GET COMMAND LINE ARGUMENTS & ERROR CHECK
    args = parser.parse_args()

    if args.makefile is None:
        print_and_exit("Please specify full path to site's makefile")
    siteMakefile = Path(args.makefile).resolve()
    if not siteMakefile.is_file():
        print_and_exit(f'{siteMakefile} is not a file')

    if args.prefix is None:
        print_and_exit('Please specify prefix')
    prefix = Path(args.prefix).resolve()

    ndim = args.dim
    if ndim is None:
        print_and_exit('Please specify problem dimension')

    # The values of these are error checked by write_library_header.py,
    # so we don't error check here.
    runtime_backend        = args.runtime
    grid_backend           = args.grid
    computation_offloading = args.offload

    configMakefile = _HOME_DIR.joinpath('Makefile.configure')
    if configMakefile.exists():
        print_and_exit(f'{configMakefile} already exists')

    print()
    print(f"Writing {configMakefile}")
    with open(configMakefile, 'w') as fptr:
        fptr.write(f"SITE_MAKEFILE = {siteMakefile}\n")
        fptr.write(f"LIB_MILHOJA_PREFIX = {prefix}\n")
        fptr.write(f"NDIM = {ndim}\n")
        if args.debug:
            fptr.write("DEBUG = true\n")
        else:
            fptr.write("DEBUG = false\n")

        fptr.write(f"FLOATING_POINT_SYSTEM = {_FLOATING_POINT_SYSTEM}\n")
        fptr.write(f"GRID_BACKEND = {grid_backend}\n")
        fptr.write(f"RUNTIME_BACKEND = {runtime_backend}\n")
        fptr.write(f"COMPUTATION_OFFLOADING = {computation_offloading}\n")
        fptr.write("THREADED_DISTRIBUTOR = false\n")

    print()

