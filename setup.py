#!/usr/bin/env python3

import argparse, sys, os, shutil
from subprocess import check_output

def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description='Runtime Setup Tool')
    parser.add_argument('--site','-s',type=str,help='site name')
    parser.add_argument('--build','-b',type=str,default='build',help='build directory')
    parser.add_argument('--test','-t',type=str,help='Name of test')
    parser.add_argument('--par','-p',type=str,help='Name of par file (in site dir)')
    parser.add_argument('--makefile','-M',type=str,help='Name of Makefile (in test dir)')
    parser.add_argument('--dim','-d',type=int,help='Dimensionality of test.')
    parser.add_argument('--debug',action="store_true",help='Set up in debug mode.')
    parser.add_argument('--coverage','-c',action="store_true",help='Enable code coverage.')
    parser.add_argument('--multithreaded',action="store_true",help='Enable multithreaded distributor.')
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

    print("Linking Makefile")
    mainMakefile = os.path.join(homeDir,'Makefile')
    os.symlink(mainMakefile,os.path.join(buildDir,'Makefile'))

    # Copy makefiles parts from site, src, and test into build dir
    print("Linking Makefile.base")
    srcMakefile = os.path.join(homeDir,'src','Makefile.base')
    os.symlink(srcMakefile,os.path.join(buildDir,'Makefile.base'))

    siteDir = os.path.join(homeDir,'sites',args.site)
    siteMakefile = os.path.join(siteDir,'Makefile.site')
    if not os.path.isfile(siteMakefile):
        raise ValueError("Site Makefile not found in site directory")
    print("Linking Makefile.site from site: "+args.site)
    os.symlink(siteMakefile,os.path.join(buildDir,'Makefile.site'))

    # Find test directory (in either test or simulations)
    testDir = os.path.join(homeDir,'test',args.test)
    if not os.path.isdir(testDir):
        testDir = os.path.join(homeDir,'simulations',args.test)
    if not os.path.isdir(testDir):
        raise ValueError("Test directory not found in test or simulations")

    # Get test makefile
    print("Linking Makefile.test from test: "+args.test)
    if args.makefile is None:
        testMakefile = os.path.join(testDir,'Makefile.test')
    else:
        testMakefile = os.path.join(testDir,args.makefile)
    if not os.path.isfile(testMakefile):
        raise ValueError("Test Makefile not found in test dir")
    os.symlink(testMakefile,os.path.join(buildDir,'Makefile.test'))

    # Write Makefile.setup in builddir
    print("Writing Makefile.setup")
    setupMakefile = os.path.join(buildDir,'Makefile.setup')
    with open(setupMakefile,'w') as f:
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
