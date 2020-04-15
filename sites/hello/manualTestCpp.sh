#!/bin/bash

# Define test problems
N_CELLS_PER_BLOCK=8
N_BLOCKS=4

MAKEFILE=Makefile_manual_cpp
BINARY=manual_cpp.x

TESTDIR=../../test

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/ManualTestCpp/constants.h
sed "s/N_CELLS_IN_X/$N_CELLS_PER_BLOCK/g" \
        $TESTDIR/constants_base.h > \
        $TESTDIR/ManualTestCpp/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_PER_BLOCK/g" $TESTDIR/ManualTestCpp/constants.h
sed -i '' "s/N_CELLS_IN_Z/1/g"                  $TESTDIR/ManualTestCpp/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/ManualTestCpp/Flash.h
sed "s/N_BLOCKS_ALONG_X/$N_BLOCKS/g" \
        $TESTDIR/Flash_base.h > \
        $TESTDIR/ManualTestCpp/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS/g" $TESTDIR/ManualTestCpp/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/1/g"         $TESTDIR/ManualTestCpp/Flash.h

# Build test binary
make -f $MAKEFILE clean all
if [[ $? -ne 0 ]]; then
    echo "Unable to compile n_cells=$N_CELLS_PER_BLOCK / n_blocks=$N_BLOCKS"
    exit 1;
fi

time ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 2;
fi
 
rm $TESTDIR/ManualTestCpp/Flash.h
rm $TESTDIR/ManualTestCpp/constants.h

