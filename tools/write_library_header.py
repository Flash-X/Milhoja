#!/usr/bin/env python3

"""
To obtain program usage information including detailed information regarding
command line arguments, run the script with the flag -h.
"""

import sys
import argparse

from pathlib import Path

#####----- VALID CONFIGURATION VALUES
# All strings in lowercase so that arguments can be case insensitive
_VALID_DIM     = [1, 2, 3]
_VALID_GRID    = ['amrex']
_VALID_FPS     = ['double']
_VALID_RUNTIME = ['none', 'cuda']
_VALID_OFFLOAD = ['none', 'openacc']

#####----- DEFAULT CONFIGURATION VALUES
# No restriction on case for these variables
_DEFAULT_DIM     = 3
_DEFAULT_FPS     = 'double'
_DEFAULT_RUNTIME = 'None'
_DEFAULT_GRID    = 'AMReX'
_DEFAULT_OFFLOAD = 'None'
# Not set by users
_DEFAULT_MDIM    = 3

#####----- PROGRAM USAGE INFO
_DESCRIPTION = \
    "Write high-level Milhoja library macros to the specified header file.\n" \
    "In general, the values associated with the macros will depend on the\n" \
    "specific flavor of library to be built using the header.  Milhoja\n" \
    "libraries cannot be built without creating this header file using\n" \
    "this script or manually.\n"
_FNAME_HELP = \
    'The name and full path of the header file to be created.  This script\n' \
    'will not overwrite pre-existing files.\n'
_DIM_HELP = \
    'Specify the dimensionality of the domains of problems to be solved\n' \
    'using the Milhoja libary to be built.\n' \
   f'\tValid Values: {_VALID_DIM}\n' \
   f'\tDefault: {_DEFAULT_DIM}\n'
_RUNTIME_HELP = \
    'Specify the runtime backend to be built into the library.\n' \
   f'\tValid Values: {_VALID_RUNTIME}\n' \
   f'\tDefault: {_DEFAULT_RUNTIME}\n'
_GRID_HELP = \
    'Specify the block-structured adaptive mesh refinement library to use\n' \
    'as the Grid backend.\n' \
   f'\tValid Values: {_VALID_GRID}\n' \
   f'\tDefault: {_DEFAULT_GRID}\n'
_FPS_HELP = \
    'Specify the IEEE floating point number system standard to use for all\n' \
    'real variables in the library.\n' \
   f'\tValid Values: {_VALID_FPS}\n' \
   f'\tDefault: {_DEFAULT_FPS}\n'
_OFFLOAD_HELP = \
    'Specify the tool that calling code will use to offload computation.\n' \
   f'\tValid Values: {_VALID_OFFLOAD}\n' \
   f'\tDefault: {_DEFAULT_OFFLOAD}\n'

#####----- ANSI TERMINAL COLORS
_ERROR = '\033[0;91;1m' # Bright Red/bold
#_ERROR = '\033[0;31;1m' # Red/bold
_NC    = '\033[0m'      # No Color/Not bold

if __name__ == '__main__':
    """
    Write the library header file in accord with the given command line
    arguments.
    """
    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=_DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('filename',  nargs=1, type=str,                           help=_FNAME_HELP)
    parser.add_argument('--dim',     '-d',    type=int, default=_DEFAULT_DIM,     help=_DIM_HELP)
    parser.add_argument('--runtime', '-r',    type=str, default=_DEFAULT_RUNTIME, help=_RUNTIME_HELP)
    parser.add_argument('--grid',    '-g',    type=str, default=_DEFAULT_GRID,    help=_GRID_HELP)
    parser.add_argument('--fps',     '-fp',   type=str, default=_DEFAULT_FPS,     help=_FPS_HELP)
    parser.add_argument('--offload', '-o',    type=str, default=_DEFAULT_OFFLOAD, help=_OFFLOAD_HELP)

    def print_and_exit(msg, error_code):
        print(file=sys.stderr)
        print(f'{_ERROR}ERROR: {msg}{_NC}', file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(error_code)

    #####----- GET COMMAND LINE ARGUMENTS & ERROR CHECK
    args = parser.parse_args()

    filename = Path(args.filename[0]).resolve()
    if filename.exists():
        print_and_exit(f'{filename} already exists', 1)

    ndim = args.dim
    if ndim not in _VALID_DIM:
        print_and_exit(f'Invalid dimension ({ndim})', 2)

    runtime_backend = args.runtime
    if runtime_backend.lower() not in _VALID_RUNTIME:
        print_and_exit(f'Invalid runtime backend ({runtime_backend})', 3)

    grid_backend = args.grid
    if grid_backend.lower() not in _VALID_GRID:
        print_and_exit(f'Invalid grid backend ({grid_backend})', 4)

    floating_point_system = args.fps
    if floating_point_system.lower() not in _VALID_FPS:
        msg = f'Invalid floating point system ({floating_point_system})'
        print_and_exit(msg, 5)

    computation_offloading = args.offload
    if computation_offloading.lower() not in _VALID_OFFLOAD:
        msg = f'Invalid computation offloading tool ({computation_offloading})'
        print_and_exit(msg, 6)

    #####----- GENERATE INTERMEDIATE DATA BASED ON ARGUMENTS
    # Dimension
    # Since the library doesn't do much computation, these are likely not
    # strictly necessary.  There are included to aid in writing tests.
    k1d, k2d, k3d = (1, 0, 0)
    if ndim >= 2:
        k2d = 1
    if ndim >= 3:
        k3d = 1

    # Runtime
    # A value of None means host-only
    if   runtime_backend.lower() == 'none':
        runtime_backend_macro = None
    elif runtime_backend.lower() == 'cuda':
        runtime_backend_macro = 'MILHOJA_USE_CUDA_BACKEND'
    else:
        print('PROGRAMMER LOGIC ERROR - runtime_backend')
        exit(100)

    # Grid
    if grid_backend.lower() == 'amrex':
        grid_backend_macro = 'MILHOJA_GRID_AMREX'
    else:
        print('PROGRAMMER LOGIC ERROR - grid_backend')
        exit(100)

    # FPS
    if floating_point_system.lower() == 'double':
        real_type     = 'MILHOJA_REAL_IS_DOUBLE'
        mpi_real_type = 'MPI_DOUBLE_PRECISION'
    else:
        print('PROGRAMMER LOGIC ERROR - floating_point_system')
        exit(100)

    # Offload
    # A value of None means host-only
    if   computation_offloading.lower() == 'none':
        offload_macro = None
    elif computation_offloading.lower() == 'openacc':
        offload_macro = 'MILHOJA_ENABLE_OPENACC_OFFLOAD'
    else:
        print('PROGRAMMER LOGIC ERROR - computation_offloading')
        exit(100)

    #####----- OUTPUT FLAG INFORMATION FOR RECORD
    print()
    print('-' * 80)
    print(f'Creating {filename.name} library header file')
    print(f'  Path                      {filename.parent}')
    print(f'  Domain dimension          {ndim}')
    print(f'  Floating point system     {floating_point_system}')
    print(f'  Grid backend              {grid_backend}')
    print(f'  Runtime backend           {runtime_backend}')
    print(f'  Computation Offloading    {computation_offloading}')
    print('-' * 80)
    print()

    #####----- WRITE CONTENTS
    with open(filename, 'w') as fptr:
        fptr.write( '#ifndef MILHOJA_H__\n')
        fptr.write( '#define MILHOJA_H__\n')
        fptr.write( '\n')
        fptr.write(f'#define MILHOJA_MDIM      {_DEFAULT_MDIM}\n')
        fptr.write(f'#define MILHOJA_NDIM      {ndim}\n')
        fptr.write(f'#define MILHOJA_K1D       {k1d}\n')
        fptr.write(f'#define MILHOJA_K2D       {k2d}\n')
        fptr.write(f'#define MILHOJA_K3D       {k3d}\n')
        fptr.write( '\n')
        # Ultra-defensive, ultra-paranoid sanity check
        fptr.write( '#if (MILHOJA_NDIM != 1) && (MILHOJA_NDIM != 2) && (MILHOJA_NDIM != 3)\n')
        fptr.write( '#error "MILHOJA_NDIM not in {1,2,3}"\n')
        fptr.write( '#endif\n')
        fptr.write( '\n')
        fptr.write(f'#define {real_type}\n')
        fptr.write(f'#define MILHOJA_MPI_REAL      {mpi_real_type}\n')
        fptr.write( '\n')
        fptr.write(f'#define {grid_backend_macro}\n')
        if runtime_backend_macro != None:
            # Rather than define the macro here, we prefer that the build system
            # define the macro at the global level.  This avoids subtle errors
            # where this header wasn't included but should have been or errors
            # in which the header must be included before a preprocessor
            # command.
            #
            # Under the assumption that calling code will include this header in
            # at least one of its source code files, the calling code build will
            # fail if they haven't chosen to build with the matching backend.
            # It's important that they do so as one of their files could include
            # in a Milhoja header that uses this macro.
            msg  = f'Milhoja library built with {runtime_backend_macro}.  '
            msg +=  'All calling code must also define this macro.'
            fptr.write( '\n')
            fptr.write(f'#ifndef {runtime_backend_macro}\n')
            fptr.write(f'#error "{msg}"\n')
            fptr.write( '#endif\n')
        if offload_macro != None:
            #  As for runtime_backend_macro
            msg  = f'Milhoja library built with {offload_macro}.  '
            msg +=  'All calling code must also define this macro.'
            fptr.write( '\n')
            fptr.write(f'#ifndef {offload_macro}\n')
            fptr.write(f'#error "{msg}"\n')
            fptr.write( '#endif\n')
        fptr.write( '\n')
        fptr.write( '#endif\n\n')

    exit(0)

