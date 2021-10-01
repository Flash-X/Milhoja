#!/usr/bin/env python3

# This script is the main workhorse of the build system. Users should invoke this script with a setup line
# similar to the following, which will set up a build directory with the necessary files for making a test.
# `python setup.py -s Thomass-MBP -d 2 -p grid_2D.par -t Grid`
#
# To make the test, cd into the build directory and run `make` or `make all`. Then, the test can be run with
# `make test` and the code coverage report can be generated with `make coverage`.
#
# To get a summary of all the command line options, run `python setup.py --help`.

import argparse, sys, os, shutil
from subprocess import check_output

def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description='Runtime Setup Tool')
    parser.add_argument('--site','-s',type=str,help='site name')
    parser.add_argument('--library','-l',type=str,help='Path to prebuilt Runtime library')
    parser.add_argument('--build','-b',type=str,default='build',help='build directory')
    parser.add_argument('test',type=str,help='Name of test')
    parser.add_argument('--par','-p',type=str,help='Name of par file (in site dir)')
    parser.add_argument('--makefile','-M',type=str,help='Name of Makefile (in site dir)')
    parser.add_argument('--dim','-d',type=int,help='Dimensionality of test.')
    parser.add_argument('--debug',action="store_true",help='Set up in debug mode.')
    parser.add_argument('--coverage','-c',action="store_true",help='Enable code coverage.')
    parser.add_argument('--multithreaded',action="store_true",help='Enable multithreaded distributor.')
    parser.add_argument('--prefix',type=str,help='Where to install Runtime library')
    args = parser.parse_args()

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

    print("Linking Makefile.F_interface")
    interfaceMakefile = os.path.join(homeDir,'interfaces','Makefile.F_interface')
    os.symlink(interfaceMakefile,os.path.join(buildDir,'Makefile.F_interface'))

    print("Linking Makefile.F_rules")
    fRulesMakefile = os.path.join(homeDir,'interfaces','Makefile.F_rules')
    os.symlink(fRulesMakefile,os.path.join(buildDir,'Makefile.F_rules'))

    siteDir = os.path.join(homeDir,'sites',args.site)
    if args.makefile is None:
        siteMakefile = os.path.join(siteDir,'Makefile.site')
    else:
        siteMakefile = os.path.join(siteDir,args.makefile)
    if not os.path.isfile(siteMakefile):
        raise ValueError("Site Makefile not found in site directory")
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


    # Copy par file into build dir
    if args.par is not None:
        print("Copying par file "+args.par+" as Flash_par.h")
        parFile = os.path.join(siteDir,args.par)
        shutil.copy(parFile,os.path.join(buildDir,'Flash_par.h'))

    # Copy Flash_ND.h and constants_ND.h to build dir as Flash.h and constants.h
    flashH = os.path.join(testDir,'Flash_{}D.h'.format(args.dim))
    constantsH = os.path.join(testDir,'constants_{}D.h'.format(args.dim))
    if os.path.isfile(flashH):
        print("Copying Flash_{}D.h as Flash.h".format(args.dim))
        shutil.copy(flashH,os.path.join(buildDir,'Flash.h'))
    if os.path.isfile(constantsH):
        print("Copying constants_{}D.h as constants.h".format(args.dim))
        shutil.copy(constantsH,os.path.join(buildDir,'constants.h'))

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
        if os.path.isfile(flashH):
            f.write('Flash.h copied from: {}\n'.format(
                    os.path.abspath(flashH)) )
        if os.path.isfile(constantsH):
            f.write('constants.h copied from: {}\n'.format(
                    os.path.abspath(constantsH)) )
        f.write('\n')

        f.write('Contents of Makefile.setup:\n')
        f.write('----------------------------\n')
        with open(setupMakefile,'r') as fread:
            for line in fread:
                f.write(line)
        f.write('----------------------------\n\n')

        f.write('Repository status:\n')
        f.write('----------------------------\n')
        f.write( check_output(['git','status']).decode('utf-8') )
        f.write('----------------------------\n\n')

        f.write('Most recent commits:\n')
        f.write('----------------------------\n')
        f.write( check_output(['git','log','--max-count','5']).decode('utf-8') )
        f.write('----------------------------\n')

    print("Successfully set up build directory!")


if __name__ == '__main__':
    main()
