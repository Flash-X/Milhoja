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

module purge
module load intel/19.0
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

TESTDIR=../../test
MAKEFILE=Makefile_threadteam_cpp
BINARY=test_threadteam_cpp.x 
DEBUG_BINARY=test_threadteam_cpp_debug.x

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/constants.h
sed "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" \
        $TESTDIR/constants_base.h > \
        $TESTDIR/constants.h
sed -i "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/constants.h
sed -i "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/Flash.h
sed "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" \
        $TESTDIR/Flash_base.h > \
        $TESTDIR/Flash.h
sed -i "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/Flash.h
sed -i "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/Flash.h

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
# Build test binary
make -f $MAKEFILE clean all
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 3;
fi
mv $BINARY ./binaries

rm $TESTDIR/Flash.h
rm $TESTDIR/constants.h

echo ""
ls -lah ./binaries/$BINARY
ldd ./binaries/$BINARY
echo ""
ls -lah ./binaries/$DEBUG_BINARY
ldd ./binaries/$DEBUG_BINARY

