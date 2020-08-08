#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Define test problem - we need dx=dy
N_CELLS_IN_X=16
N_CELLS_IN_Y=16
N_CELLS_IN_Z=16

N_BLOCKS_X=4
N_BLOCKS_Y=4
N_BLOCKS_Z=4

N_DIMS=2
LREFINE=3


cd ..

# Specified relative to location of Makefile
TESTDIR=../../test
MAKEFILE=Makefile_adv_cpp
BINARY=test_adv_cpp.x 
DEBUG_BINARY=test_adv_cpp_debug.x

# Setup constants.h with current simulation's Grid parameters
rm $TESTDIR/Adv/constants.h
cp $TESTDIR/constants_base.h $TESTDIR/Adv/constants.h
sed -i '' "s/N_CELLS_IN_X/$N_CELLS_IN_X/g" $TESTDIR/Adv/constants.h
sed -i '' "s/N_CELLS_IN_Y/$N_CELLS_IN_Y/g" $TESTDIR/Adv/constants.h
sed -i '' "s/N_CELLS_IN_Z/$N_CELLS_IN_Z/g" $TESTDIR/Adv/constants.h
sed -i '' "s/N_DIMENSIONS/$N_DIMS/g" $TESTDIR/Adv/constants.h

# Setup Flash.h with current simulation's Grid parameters
rm $TESTDIR/Adv/Flash.h
cp $TESTDIR/Flash_base.h $TESTDIR/Adv/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_X/$N_BLOCKS_X/g" $TESTDIR/Adv/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Y/$N_BLOCKS_Y/g" $TESTDIR/Adv/Flash.h
sed -i '' "s/N_BLOCKS_ALONG_Z/$N_BLOCKS_Z/g" $TESTDIR/Adv/Flash.h
sed -i '' "s/REFINEMENT_LEVELS/$LREFINE/g" $TESTDIR/Adv/Flash.h
sed -i '' "s/NUNKVAR    2/NUNKVAR 1/g" $TESTDIR/Adv/Flash.h
#sed -i '' "s/NGUARD 1/NGUARD 0/g" $TESTDIR/Adv/Flash.h

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

rm $TESTDIR/Adv/Flash.h
rm $TESTDIR/Adv/constants.h

