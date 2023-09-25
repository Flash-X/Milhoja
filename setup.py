#!/usr/bin/env python3

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
_DEFAULT_BUILD    = 'build'
_DEFAULT_MAKEFILE = 'Makefile.site'
# For no apparent reason, default is host-only execution.
_DEFAULT_GRID     = 'AMReX'
_DEFAULT_RUNTIME  = 'None'
_DEFAULT_OFFLOAD  = 'None'

#####----- PROGRAM USAGE INFO
_DESCRIPTION = \
    "This script is the main workhorse of the build system. Users should\n" \
    "invoke this script with a setup line similar to the following, which\n" \
    "will set up a build directory with the necessary files for making a\n" \
    "test.\n\n" \
    "\tsetup.py Grid -s summit -d 2 -p grid_2D.json\n\n" \
    "The build directory is always created in the root folder of the called\n" \
    "script's repository.  If a file or directory already exists with that\n" \
    "name, this script deletes it without warning so that each build is clean.\n\n" \
    "To make the test, cd into the build directory and run 'make' or\n" \
    "'make all'. Then, the test can be run with 'make test' and the code\n" \
    "coverage report can be generated with 'make coverage'.\n"
_BUILD_HELP = \
    'Name of desired build directory\n' \
   f'\tDefault: {_DEFAULT_BUILD}\n'
_MAKEFILE_HELP = \
    'Name of Makefile (in site dir)\n' \
   f'\tDefault: {_DEFAULT_MAKEFILE}\n'
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

#####----- HARDCODED VARIABLES
# setup.py is located in the repository root directory.  We assemble the build
# relative to that directory.
_HOME_DIR = Path(__file__).resolve().parent

# If a test directory contains this file, we assume that it must be run to
# automatically generate all code
_CODE_GENERATION_SCRIPT = "generate_code_for_milhoja.sh"

PAR_FILENAME_BASE = 'RuntimeParameters'

if __name__ == '__main__':
    """
    Setup a build directory in accord with the given command line arguments.
    """
    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=_DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('test',             type=str,                            help='Name of test')
    parser.add_argument('--site',     '-s', type=str,                            help='[mandatory] site name')
    parser.add_argument('--library',  '-l', type=str,                            help='[mandatory] Path to prebuilt static library')
    parser.add_argument('--build',    '-b', type=str, default=_DEFAULT_BUILD,    help=_BUILD_HELP)
    parser.add_argument('--par',      '-p', type=str,                            help='[mandatory] Name of par file (in site dir)')
    parser.add_argument('--makefile', '-M', type=str, default=_DEFAULT_MAKEFILE, help=_MAKEFILE_HELP)
    parser.add_argument('--dim',      '-d', type=int,                            help='[mandatory] Dimensionality of test.')
    parser.add_argument('--runtime',  '-r', type=str, default=_DEFAULT_RUNTIME,  help=_RUNTIME_HELP)
    parser.add_argument('--grid',     '-g', type=str, default=_DEFAULT_GRID,     help=_GRID_HELP)
    parser.add_argument('--offload',  '-o', type=str, default=_DEFAULT_OFFLOAD,  help=_OFFLOAD_HELP)
    parser.add_argument('--debug',         action="store_true", help='Set up in debug mode.')
    parser.add_argument('--coverage','-c', action="store_true", help='Enable code coverage.')
    parser.add_argument('--multithreaded', action="store_true", help='Enable multithreaded distributor.')

    def print_and_exit(msg):
        print(file=sys.stderr)
        print(f'{_ERROR}SETUP ERROR: {msg}{_NC}', file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(1)

    #####----- GET COMMAND LINE ARGUMENTS & ERROR CHECK
    args = parser.parse_args()

    if args.site is None:
        print_and_exit('Please specify a site')
    siteDir = _HOME_DIR.joinpath('sites', args.site)
    if not siteDir.is_dir():
        print_and_exit(f'Site directory {siteDir} does not exist')

    # OK to overwrite previous build
    buildDir = _HOME_DIR.joinpath(args.build)
    if buildDir.is_dir():
        shutil.rmtree(buildDir)

    test_name = args.test
    testDir = _HOME_DIR.joinpath('test')
    path = test_name.split('/')
    for each in path:
        testDir = testDir.joinpath(each)
    if not testDir.is_dir():
        print_and_exit(f'Test directory {testDir} does not exist')

    if args.library is None:
        print_and_exit('Please specify library path')
    libraryDir = Path(args.library).resolve()
    if not libraryDir.is_dir():
        print_and_exit(f'Library path {libraryDir} does not exist')

    ndim = args.dim
    if ndim is None:
        print_and_exit('Please specify problem dimension')

    # The values of these are error checked by write_library_header.py,
    # so we don't error check here.
    runtime_backend        = args.runtime
    grid_backend           = args.grid
    computation_offloading = args.offload

    par_filename_src = args.par
    if par_filename_src is None:
        print_and_exit('Please specify a parameter file')
    parFile_src  = siteDir.joinpath(par_filename_src)
    if not parFile_src.is_file():
        print_and_exit(f'{parFile_src} is not a file')

    #####----- ASSEMBLE BUILD FOLDER & CONTENTS
    print("Orchestration Runtime setup")
    print("---------------------------")

    ##-- MAKE BUILD DIRECTORY
    # Make in root of repo
    print(f"Creating build directory: {buildDir.name}")
    os.makedirs(buildDir)

    ##-- MAIN MAKEFILE
    print("Copying Makefile")
    mainMakefile_src  = _HOME_DIR.joinpath('Makefile.base')
    mainMakefile_dest =  buildDir.joinpath('Makefile')
    assert(not mainMakefile_dest.exists())
    shutil.copy(mainMakefile_src, mainMakefile_dest)

    ##-- SITE MAKEFILE
    siteMakefile_src  =  siteDir.joinpath(args.makefile)
    siteMakefile_dest = buildDir.joinpath('Makefile.site')
    if not siteMakefile_src.is_file():
        print_and_exit(f'{siteMakefile_src} is not a file')
    assert(not siteMakefile_dest.exists())
    print(f"Copying {siteMakefile_src.name} for site {siteDir.name}")
    shutil.copy(siteMakefile_src, siteMakefile_dest)

    ##-- TEST MAKEFILE
    print(f"Copying Makefile.test from test {test_name}")
    testMakefile_src  =  testDir.joinpath('Makefile.test')
    testMakefile_dest = buildDir.joinpath('Makefile.test')
    if not testMakefile_src.is_file():
        print_and_exit(f'{testMakefile_src} is not a file')
    assert(not testMakefile_dest.exists())
    shutil.copy(testMakefile_src, testMakefile_dest)

    ##-- CHECK JSON file
    print(f"Checking JSON file in {test_name}")
    jsonPath = testDir.joinpath(f'{os.path.basename(test_name)}.json')
    if not jsonPath.is_file():
        print(f"Generating empty json {jsonPath}")
        with open(jsonPath, "w") as fptr:
    	    fptr.write("{\n}")

    ##-- GENERATE Makefile.setup
    print("Writing Makefile.setup")
    setupMakefile = buildDir.joinpath('Makefile.setup')
    assert(not setupMakefile.exists())
    with open(setupMakefile, 'w') as fptr:
        fptr.write("ifneq ($(BASEDIR),{})\n".format(_HOME_DIR) )
        fptr.write("$(warning BASEDIR=$(BASEDIR) but repository root directory is {})\n".format(_HOME_DIR))
        fptr.write("endif\n\n")
        fptr.write("BUILDDIR = $(BASEDIR)/{}\n".format(args.build))
        fptr.write("GENDIR=$(BASEDIR)/tools/datapacket_generator\n")
        if args.debug:
            fptr.write(f"DEBUG = true\n")
        else:
            fptr.write("DEBUG = false\n")

        if args.coverage:
            fptr.write("CODECOVERAGE = true\n")
        else:
            fptr.write("CODECOVERAGE = false\n")

        fptr.write(f"NDIM = {ndim}\n")

        if args.multithreaded:
            fptr.write("THREADED_DISTRIBUTOR = true\n")
        else:
            fptr.write("THREADED_DISTRIBUTOR = false\n")

        if runtime_backend.lower() == 'cuda':
            fptr.write("USE_CUDA_BACKEND = true\n")
        else:
            fptr.write("USE_CUDA_BACKEND = false\n")

        if computation_offloading.lower() == 'openacc':
            fptr.write("ENABLE_OPENACC_OFFLOAD = true\n")
        else:
            fptr.write("ENABLE_OPENACC_OFFLOAD = false\n")

        fptr.write("\n")
        fptr.write("LIB_MILHOJA = {}\n".format(libraryDir))

    ##-- DERIVE PAR FILENAME & COPY TO BUILD DIR
    # The final filename will have the same extension of the given file.
    # A file that ends in
    #  * .json is ready for immediate use
    #  * .json_base needs updating before use
    tmp = par_filename_src.split('.')
    if (len(tmp) != 2) or (tmp[1] not in ['json', 'json_base']):
        print_and_exit('Par file names must be of the form <name>.json[_base]')
    _, par_ext = tmp

    parFile_dest = buildDir.joinpath(f'{PAR_FILENAME_BASE}.{par_ext}')
    assert(not parFile_dest.exists())
    print(f"Copying {parFile_src.name} as {parFile_dest.name}")
    shutil.copy(parFile_src, parFile_dest)

    ##-- Log setup info & build metadata
    print("Writing setup.log")
    logfileName = buildDir.joinpath("setup.log")
    with open(logfileName,'w') as fptr:
        fptr.write('Setup command line: \n')
        fptr.write(f'{Path(sys.argv[0]).resolve()}')
        fptr.write(' ')
        fptr.write(' '.join(sys.argv[1:]))
        fptr.write('\n\n\n')

        fptr.write('Build directory: \n')
        fptr.write(f'{buildDir}')
        fptr.write('\n\n')

        fptr.write('Path to copied files:\n')
        fptr.write(f'Makefile            -->    {mainMakefile_src}\n')
        fptr.write(f'Makefile.site       -->    {siteMakefile_src}\n')
        fptr.write(f'Makefile.test       -->    {testMakefile_src}\n')
        fptr.write(f'{parFile_dest.name} -->    {parFile_src}\n')
        fptr.write('\n')

    #####----- GENERATE CODE FOR MILHOJA
    generator = testDir.joinpath(_CODE_GENERATION_SCRIPT)
    genpath = buildDir.joinpath("generated_code")
    generatedMakefile = genpath.joinpath("..", "Makefile.generated")
    if generator.is_file():
        assert not genpath.exists()
        os.mkdir(genpath)
        # TODO: Pass makefile name as well
        # TODO: Check error code of shell call
        sbp.call([str(generator), str(genpath)])
    else:
        print(f"No automatic code generation required by test")
        with open(generatedMakefile, "w") as fptr:
            fptr.write("")

    #####----- SET PERMISSIONS TO READ-ONLY
    os.chmod(mainMakefile_dest, 0o444)
    os.chmod(siteMakefile_dest, 0o444)
    os.chmod(testMakefile_dest, 0o444)
    os.chmod(generatedMakefile, 0o444)
    os.chmod(setupMakefile,     0o444)
    os.chmod(parFile_dest,      0o444)
    os.chmod(logfileName,       0o444)

    print("Successfully set up build directory!")

