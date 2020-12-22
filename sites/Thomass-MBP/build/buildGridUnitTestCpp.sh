#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Define test problem - we need dx=dy
N_CELLS_IN_X=8
N_CELLS_IN_Y=8
N_CELLS_IN_Z=1

N_BLOCKS_X=4
N_BLOCKS_Y=4
N_BLOCKS_Z=1

N_DIMS=2
LREFINE=3


rm -r test_plt_0000*

# Specified relative to location of Makefile
TESTDIR=../test
MAKEFILE=Makefile_grid_cpp
BINARY=test_grid_cpp.x 
DEBUG_BINARY=test_grid_cpp_debug.x

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/Grid/constants.h
cp $TESTDIR/constants_base.h $TESTDIR/Grid/constants.h
sed -i '' "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" $TESTDIR/Grid/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/Grid/constants.h
sed -i '' "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/Grid/constants.h
sed -i '' "s/N_DIMENSIONS/$N_DIMS/g" $TESTDIR/Grid/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/Grid/Flash.h
cp $TESTDIR/Flash_base.h $TESTDIR/Grid/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" $TESTDIR/Grid/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/Grid/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/Grid/Flash.h
sed -i '' "s/REFINEMENT_LEVELS/$LREFINE/g" $TESTDIR/Grid/Flash.h

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

rm $TESTDIR/Grid/Flash.h
rm $TESTDIR/Grid/constants.h

