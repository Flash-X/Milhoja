#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Define test problem - we need dx=dy
N_CELLS_IN_X=8
N_CELLS_IN_Y=16
N_CELLS_IN_Z=1

N_BLOCKS_X=256
N_BLOCKS_Y=128
N_BLOCKS_Z=1

. $RUNTIME_PGI_SETUP

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
echo
date
echo
module list
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
MAKEFILE=Makefile_runtime_cuda
BINARY=test_runtime_cuda.x
DEBUG_BINARY=test_runtime_cuda_debug.x

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/Runtime/constants.h
sed "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" \
        $TESTDIR/constants_base.h > \
        $TESTDIR/Runtime/constants.h
sed -i "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/Runtime/constants.h
sed -i "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/Runtime/constants.h
sed -i "s/N_DIMENSIONS/2/g"             $TESTDIR/Runtime/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/Runtime/Flash.h
sed "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" \
        $TESTDIR/Flash_base.h > \
        $TESTDIR/Runtime/Flash.h
sed -i "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/Runtime/Flash.h
sed -i "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/Runtime/Flash.h
sed -i "s/REFINEMENT_LEVELS/1/g"          $TESTDIR/Runtime/Flash.h

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

rm $TESTDIR/Runtime/Flash.h
rm $TESTDIR/Runtime/constants.h

echo ""
ls -lah ./binaries/$BINARY
ldd ./binaries/$BINARY
echo ""
ls -lah ./binaries/$DEBUG_BINARY
echo ""
ldd ./binaries/$DEBUG_BINARY

