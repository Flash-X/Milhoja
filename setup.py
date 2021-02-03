#!/usr/bin/env python3

import argparse, sys, os, shutil


def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description='Runtime Setup Tool')
    parser.add_argument('--site','-s',type=str,help='site name')
    parser.add_argument('--build','-b',type=str,default='build',help='build directory')
    parser.add_argument('--test','-t',type=str,help='Name of test')
    parser.add_argument('--par','-p',type=str,help='Name of par file (in site dir)')
    parser.add_argument('--debug','-d',action="store_true",help='Set up in debug mode.')
    parser.add_argument('--coverage','-c',action="store_true",help='Enable code coverage.')
    args = parser.parse_args()

    # Setup.py is located in the repository root directory.
    homeDir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Make build directory in root directory. Delete it first if it already exists.
    buildDir = os.path.join( homeDir, args.build)
    if os.path.isdir(buildDir):
        shutil.rmtree(buildDir)
    os.makedirs(buildDir)

    # Copy makefiles parts from site, src, and test into build dir
    siteDir = os.path.join(homeDir,'sites',args.site)
    siteMakefile = os.path.join(siteDir,'Makefile.site')
    shutil.copy(siteMakefile,buildDir)

    srcMakefile = os.path.join(homeDir,'src','Makefile.base')
    shutil.copy(srcMakefile,buildDir)

    testMakefile = os.path.join(homeDir,'test',args.test,'Makefile.test')
    shutil.copy(testMakefile,buildDir)

    mainMakefile = os.path.join(homeDir,'Makefile')
    shutil.copy(mainMakefile,buildDir)

    # Write Makefile.setup in builddir
    setupMakefile = os.path.join(buildDir,'Makefile.setup')
    with open(setupMakefile,'w') as f:
        f.write("BUILDDIR = $(BASEDIR)/{}\n".format(args.build))
        f.write("OBJDIR = $(BUILDDIR)/obj\n")
        if args.debug:
            f.write("DEBUG = true\n")
        else:
            f.write("DEBUG = false\n")

        if args.coverage:
            f.write("CODECOVERAGE = true\n")
        else:
            f.write("CODECOVERAGE = false\n")


    # Copy par file into build dir
    #(Start with: copy Flash.h, constants.h)
    #for demo, copy build file
    buildfile = os.path.join(siteDir,'build',args.par)
    shutil.copy(buildfile,buildDir)




if __name__ == '__main__':
    main()
