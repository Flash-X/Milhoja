#!/usr/bin/env python3

import argparse, sys, os, shutil


def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description='Runtime Setup Tool')
    parser.add_argument('--site',type=str,help='site name')
    parser.add_argument('--build',type=str,default='build',help='build directory')
    parser.add_argument('--test',type=str,help='Name of test')
    parser.add_argument('--par',type=str,help='Name of par file (in site dir)')
    args = parser.parse_args()

    # Setup.py is located in repo home directory
    homeDir = os.path.dirname(os.path.abspath(sys.argv[0]))

    #1. Make build directory
    buildDir = os.path.join( homeDir, args.build)
    if not os.path.isdir(buildDir): os.makedirs(buildDir)

    #2. Copy makefiles parts from site, src, and test into build dir
    siteDir = os.path.join(homeDir,'sites',args.site)
    siteMakefile = os.path.join(siteDir,'Makefile.site')
    shutil.copy(siteMakefile,buildDir)

    srcMakefile = os.path.join(homeDir,'src','Makefile.base')
    shutil.copy(srcMakefile,buildDir)

    testMakefile = os.path.join(homeDir,'test',args.test,'Makefile.test')
    shutil.copy(testMakefile,buildDir)

    mainMakefile = os.path.join(homeDir,'Makefile')
    shutil.copy(mainMakefile,buildDir)

    #for demo, copy build file
    buildfile = os.path.join(siteDir,'build','buildGridUnitTestCpp.sh')
    shutil.copy(buildfile,buildDir)

    #4. Copy par file into build dir
    #(Start with: copy Flash.h, constants.h)




if __name__ == '__main__':
    main()
