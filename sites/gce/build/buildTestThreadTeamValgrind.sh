#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
module purge
module load gcc/8.3.0-fjpc5ys
module list

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
date
echo
echo "Current Branches in Repository"
echo "-----------------------------------------------------------"
git branch -vva
echo
echo "Last Git repository log entries"
echo "-----------------------------------------------------------"
git log --oneline -10
echo
echo "Current state of the local workspace"
echo "-----------------------------------------------------------"
git status

# Run from location of Makefile
cd ..

# Specified relative to location of Makefile
TESTDIR=../../test
MAKEFILE=Makefile_threadteam_valgrind
BINARY=test_threadteam_valgrind.x

# Build non-debug mode
echo ""
echo "Building production version"
echo "----------------------------------------------------------"
make -f $MAKEFILE clean all
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 4;
fi
mv $BINARY ./binaries

echo ""
ls -lah ./binaries/$BINARY
ldd ./binaries/$BINARY
echo ""

