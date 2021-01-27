#!/bin/bash

# Define test problem - we need dx=dy
N_CELLS_IN_X=8
N_CELLS_IN_Y=16
N_CELLS_IN_Z=1

N_BLOCKS_X=256
N_BLOCKS_Y=128
N_BLOCKS_Z=1

N_THREADS_PER_TEAM=4

MAKEFILE=Makefile_threadteam_cpp
BINARY=test_threadteam_cpp.x 

TESTDIR=../../test

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/constants.h
sed "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" \
        $TESTDIR/constants_base.h > \
        $TESTDIR/ThreadTeam/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/ThreadTeam/constants.h
sed -i '' "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/ThreadTeam/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/Flash.h
sed "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" \
        $TESTDIR/Flash_base.h > \
        $TESTDIR/ThreadTeam/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/ThreadTeam/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/ThreadTeam/Flash.h

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
rm $TESTDIR/Flash.h
rm $TESTDIR/constants.h

time ./$BINARY $N_THREADS_PER_TEAM
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 4;
fi

