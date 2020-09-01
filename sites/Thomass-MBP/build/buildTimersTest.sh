#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Define test problem - we need dx=dy
N_CELLS_IN_X=1
N_CELLS_IN_Y=1
N_CELLS_IN_Z=1

N_BLOCKS_X=1
N_BLOCKS_Y=1
N_BLOCKS_Z=1

N_DIMS=1
LREFINE=1


cd ..

# Specified relative to location of Makefile
TESTDIR=../../test
MAKEFILE=Makefile_timers_cpp
BINARY=test_timers_cpp.x 
DEBUG_BINARY=test_timers_cpp_debug.x

# Setup constants.h with current simulation's Timers parameters
rm $TESTDIR/Timers/constants.h
cp $TESTDIR/constants_base.h $TESTDIR/Timers/constants.h
sed -i '' "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" $TESTDIR/Timers/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/Timers/constants.h
sed -i '' "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/Timers/constants.h
sed -i '' "s/N_DIMENSIONS/$N_DIMS/g" $TESTDIR/Timers/constants.h

# Setup Flash.h with current simulation's Timers parameters
rm $TESTDIR/Timers/Flash.h
cp $TESTDIR/Flash_base.h $TESTDIR/Timers/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" $TESTDIR/Timers/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/Timers/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/Timers/Flash.h
sed -i '' "s/REFINEMENT_LEVELS/$LREFINE/g" $TESTDIR/Timers/Flash.h

# Build debug mode
#echo ""
#echo "Building debug version"
#echo "----------------------------------------------------------"
#make -f $MAKEFILE clean all DEBUG=T
#if [[ $? -ne 0 ]]; then
#    echo "Unable to compile $BINARY"
#    exit 3;
#fi
#mv $BINARY ./binaries/$DEBUG_BINARY

# Build non-debug mode
echo ""
echo "Building production version"
echo "----------------------------------------------------------"
make -f $MAKEFILE clean all GLOG=T
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 4;
fi
mv $BINARY ./binaries/$BINARY

rm $TESTDIR/Timers/Flash.h
rm $TESTDIR/Timers/constants.h

