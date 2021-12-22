#!/usr/bin/env python3

"""
To obtain program usage information including detailed information regarding
command line arguments, run the script with the flag -h.
"""

import os
import sys
import shutil
import argparse

import subprocess as sbp

from pathlib import Path

#####----- FIXED CONFIGURATION VALUES
# Always build with IEEE double precision reals.
_FLOATING_POINT_SYSTEM = 'double'

#####----- DEFAULT CONFIGURATION VALUES
_DEFAULT_BUILD   = 'build'
# For no apparent reason, default is host-only execution.
_DEFAULT_GRID    = 'AMReX'
_DEFAULT_RUNTIME = 'None'
_DEFAULT_OFFLOAD = 'None'

#####----- PROGRAM USAGE INFO
_DESCRIPTION = \
    "This script is the main workhorse of the build system. Users should\n" \
    "invoke this script with a setup line similar to the following, which\n" \
    "will set up a build directory with the necessary files for making a\n" \
    "test.\n\n" \
    "\tpython setup.py -s Thomass-MBP -d 2 -p grid_2D.par Grid\n\n" \
    "The build directory is always created in the root folder of the called\n" \
    "script's repository.  If a file or directory already exists with that\n" \
    "name, this script deletes it without warning so that each build is clean.\n\n" \
    "To make the test, cd into the build directory and run 'make' or\n" \
    "'make all'. Then, the test can be run with 'make test' and the code\n" \
    "coverage report can be generated with 'make coverage'.\n"
_RUNTIME_HELP = \
    'Specify the runtime backend to be used by the test.  Refer to the\n' \
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
#_ERROR = '\033[0;31;1m' # Red/bold
_NC    = '\033[0m'      # No Color/Not bold

if __name__ == '__main__':
    """
    Setup a build directory in accord with the given command line arguments.
    """
    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=_DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('test',             type=str,                           help='Name of test')
    parser.add_argument('--site',     '-s', type=str,                           help='site name')
    parser.add_argument('--library',  '-l', type=str,                           help='Path to prebuilt Runtime library')
    parser.add_argument('--build',    '-b', type=str, default=_DEFAULT_BUILD,   help='build directory')
    parser.add_argument('--par',      '-p', type=str,                           help='Name of par file (in site dir)')
    parser.add_argument('--makefile', '-M', type=str,                           help='Name of Makefile (in site dir)')
    parser.add_argument('--dim',      '-d', type=int,                           help='Dimensionality of test.')
    parser.add_argument('--runtime',  '-r', type=str, default=_DEFAULT_RUNTIME, help=_RUNTIME_HELP)
    parser.add_argument('--grid',     '-g', type=str, default=_DEFAULT_GRID,    help=_GRID_HELP)
    parser.add_argument('--offload',  '-o', type=str, default=_DEFAULT_OFFLOAD, help=_OFFLOAD_HELP)
    parser.add_argument('--prefix',         type=str,                           help='Where to install Runtime library')
    parser.add_argument('--debug',         action="store_true", help='Set up in debug mode.')
    parser.add_argument('--coverage','-c', action="store_true", help='Enable code coverage.')
    parser.add_argument('--multithreaded', action="store_true", help='Enable multithreaded distributor.')

    def print_and_exit(msg, error_code):
        print(file=sys.stderr)
        print(f'{_ERROR}SETUP ERROR: {msg}{_NC}', file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(error_code)

    #####----- GET COMMAND LINE ARGUMENTS & ERROR CHECK
    args = parser.parse_args()

    # The values of these are error checked by write_library_header.py,
    # so we don't error check here.
    runtime_backend        = args.runtime
    grid_backend           = args.grid
    computation_offloading = args.offload

    #####----- ASSEMBLE BUILD FOLDER & CONTENTS
    print("Orchestration Runtime setup")
    print("---------------------------")

    # Setup.py is located in the repository root directory.
    homeDir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Make build directory in root directory. Delete it first if it already exists.
    buildDir = os.path.join( homeDir, args.build)
    print("Creating build directory: "+args.build)
    if os.path.isdir(buildDir):
        shutil.rmtree(buildDir)
    os.makedirs(buildDir)

    # Link main makefile
    print("Linking Makefile")
    mainMakefile = os.path.join(homeDir,'Makefile')
    os.symlink(mainMakefile,os.path.join(buildDir,'Makefile'))

    # Link makefiles parts from site and src
    print("Linking Makefile.base")
    srcMakefile = os.path.join(homeDir,'src','Makefile.base')
    os.symlink(srcMakefile,os.path.join(buildDir,'Makefile.base'))

    siteDir = os.path.join(homeDir,'sites',args.site)
    if args.makefile is None:
        siteMakefile = os.path.join(siteDir,'Makefile.site')
    else:
        siteMakefile = os.path.join(siteDir,args.makefile)
    if not os.path.isfile(siteMakefile):
        raise ValueError(f"Site Makefile {siteMakefile} not found in site directory")
    print("Linking Makefile.site from site: "+args.site)
    os.symlink(siteMakefile,os.path.join(buildDir,'Makefile.site'))

    # Find test directory
    testDir = os.path.join(homeDir,'test',args.test)
    if not os.path.isdir(testDir):
        raise ValueError("Test directory not found")

    if (args.test == 'library'):
        if not args.prefix:
            raise ValueError("Need to supply prefix if building library!")
    else:
        # Get test makefile
        print("Linking Makefile.test from test: "+args.test)
        testMakefile = os.path.join(testDir,'Makefile.test')
        if not os.path.isfile(testMakefile):
            raise ValueError("Test Makefile not found in test dir")
        os.symlink(testMakefile,os.path.join(buildDir,'Makefile.test'))

    # Write Makefile.setup in builddir
    print("Writing Makefile.setup")
    setupMakefile = os.path.join(buildDir,'Makefile.setup')
    with open(setupMakefile,'w') as f:
        f.write("ifneq ($(BASEDIR),{})\n".format(homeDir) )
        f.write("$(warning BASEDIR=$(BASEDIR) but repository root directory is {})\n".format(homeDir))
        f.write("endif\n\n")
        f.write("BUILDDIR = $(BASEDIR)/{}\n".format(args.build))
        if args.debug:
            f.write("DEBUG = true\n")
        else:
            f.write("DEBUG = false\n")

        if args.coverage:
            f.write("CODECOVERAGE = true\n")
        else:
            f.write("CODECOVERAGE = false\n")

        f.write("NDIM = {}\n".format(args.dim))
        if args.multithreaded:
            f.write("THREADED_DISTRIBUTOR = true\n")
        else:
            f.write("THREADED_DISTRIBUTOR = false\n")

        if runtime_backend.lower() == 'cuda':
            f.write("USE_CUDA_BACKEND = true\n")
        else:
            f.write("USE_CUDA_BACKEND = false\n")

        if computation_offloading.lower() == 'openacc':
            f.write("ENABLE_OPENACC_OFFLOAD = true\n")
        else:
            f.write("ENABLE_OPENACC_OFFLOAD = false\n")

        f.write("\n")
        if args.test == 'library':
            f.write("LIBONLY = True\n")
            f.write("LIB_RUNTIME_PREFIX = {}\n".format(args.prefix))
        else:
            f.write("# Leave blank if building a test\n")
            f.write("LIBONLY = \n")
            if args.library is not None:
                f.write("LINKLIB = True\n")
                f.write("LIB_RUNTIME = {}\n".format(args.library))
            else:
                f.write("# Leave blank if not linking a prebuilt library!\n")
                f.write("LINKLIB = \n")
                f.write("# Should be current dir (i.e. `.`) if not linking prebuilt library\n")
                f.write("LIB_RUNTIME = .")

    ##-- Construct Milhoja.h file in build dir
    fname_header = Path(buildDir).resolve().joinpath('Milhoja.h')
    fname_script = Path(homeDir).resolve().joinpath('tools', 'write_library_header.py')
    if not fname_script.is_file():
        print_and_exit(f'Cannot find {fname_script}', 1)

    # Since the build folder is always built new by this script, we are certain
    # that the header file does not already exist.
    #
    # Specify all flags so that the defaults defined in this script are used.
    cmd = [str(fname_script),
           str(fname_header),
           '--dim',     str(args.dim),
           '--runtime', runtime_backend,
           '--grid',    grid_backend,
           '--fps',     _FLOATING_POINT_SYSTEM,
           '--offload', computation_offloading]
    print('Creating Milhoja.h header file')
    try:
        # Store stdout output for later logging
        hdr_stdout = sbp.check_output(cmd).decode('utf-8')
    except sbp.CalledProcessError:
        print_and_exit(f'Unable to create Milhoja.h', 2)

    # Copy par file into build dir
    if args.par is not None:
        print("Copying par file "+args.par+" as Flash_par.h")
        parFile = os.path.join(siteDir,args.par)
        shutil.copy(parFile,os.path.join(buildDir,'Flash_par.h'))

    # Write the setup logfile
    print("Writing setup.log")
    logfileName = os.path.join(buildDir,"setup.log")
    with open(logfileName,'w') as f:
        f.write('Setup command line: \n')
        f.write(os.path.abspath(sys.argv[0]))
        f.write(' ')
        f.write(' '.join(sys.argv[1:]))
        f.write('\n\n\n')

        f.write('Build directory: \n')
        f.write(os.path.abspath(buildDir) )
        f.write('\n\n')

        f.write('Path to linked files:\n')
        f.write('Makefile --> {}\n'.format(os.path.abspath(mainMakefile)))
        f.write('Makefile.base --> {}\n'.format(os.path.abspath(srcMakefile)))
        f.write('Makefile.site --> {}\n'.format(os.path.abspath(siteMakefile)))
        if(args.test != 'library'):
            f.write('Makefile.test --> {}\n'.format(os.path.abspath(testMakefile)))
        f.write('\n')

        f.write('Path to copied files:\n')
        if args.par is not None:
            f.write('Flash_par.h copied from: {}\n'.format(
                    os.path.abspath(parFile)) )
        f.write('\n')

        f.write('Contents of Makefile.setup:\n')
        f.write('----------------------------\n')
        with open(setupMakefile,'r') as fread:
            for line in fread:
                f.write(line)
        f.write('----------------------------\n\n')

        f.write(hdr_stdout)

        f.write('Repository status:\n')
        f.write('----------------------------\n')
        f.write( sbp.check_output(['git','status']).decode('utf-8') )
        f.write('----------------------------\n\n')

        f.write('Most recent commits:\n')
        f.write('----------------------------\n')
        f.write( sbp.check_output(['git','log','--max-count','5']).decode('utf-8') )
        f.write('----------------------------\n')

    print("Successfully set up build directory!")

