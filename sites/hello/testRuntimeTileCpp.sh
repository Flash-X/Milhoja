#!/bin/bash

# Define test problem - we need dx=dy
N_CELLS_IN_X=8
N_CELLS_IN_Y=16
N_CELLS_IN_Z=1

N_BLOCKS_X=256
N_BLOCKS_Y=128
N_BLOCKS_Z=1

MAKEFILE=Makefile_runtime_tile_cpp
BINARY=test_runtime_tile_cpp.x 

TESTDIR=../../test

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/RuntimeTile/constants.h
sed "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" \
        $TESTDIR/constants_base.h > \
        $TESTDIR/RuntimeTile/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/RuntimeTile/constants.h
sed -i '' "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/RuntimeTile/constants.h
sed -i '' "s/N_DIMENSIONS/2/g"             $TESTDIR/RuntimeTile/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/RuntimeTile/Flash.h
sed "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" \
        $TESTDIR/Flash_base.h > \
        $TESTDIR/RuntimeTile/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/RuntimeTile/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/RuntimeTile/Flash.h
sed -i '' "s/REFINEMENT_LEVELS/1/g"          $TESTDIR/RuntimeTile/Flash.h

# Build test binary
if   [[ "$#" -eq 0 ]]; then
        make -f $MAKEFILE clean all
elif [[ "$#" -eq 1 ]]; then
    if [[ "$1" = "--debug" ]]; then
        make -f $MAKEFILE clean all DEBUG=T
    else
        echo "Unknown command line argument", $1
        exit 1;
    fi
elif [[ "$#" -gt 1 ]]; then
    echo "At most one command line argument accepted"
    exit 2;
fi

# Confirm build and clean-up
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 3;
fi
rm $TESTDIR/RuntimeTile/Flash.h
rm $TESTDIR/RuntimeTile/constants.h

time ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 4;
fi

