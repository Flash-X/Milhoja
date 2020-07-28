#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
module purge
module load git
module load gcc/9.1.0
module load spectrum-mpi/10.3.1.2-20200121
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
MAKEFILE=Makefile_runtime_null_cpp
BINARY=test_runtime_null_cpp.x 
DEBUG_BINARY=test_runtime_null_cpp_debug.x

# Build debug mode
echo ""
echo "Building debug version"
echo "----------------------------------------------------------"
make -f $MAKEFILE clean all DEBUG=T
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 3;
fi
mv $BINARY ./binaries/$DEBUG_BINARY

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
ls -lah ./binaries/$DEBUG_BINARY
echo ""
ldd ./binaries/$DEBUG_BINARY

