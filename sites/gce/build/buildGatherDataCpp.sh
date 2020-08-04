#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Define test problems
N_CELLS_PER_BLOCK=(8 16 32 64 128)
N_BLOCKS=(2 4 8 16 32 64)

module purge
module load intel/19.0
module load mpich/3.3.2-intel
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
MAKEFILE=Makefile_gatherData_cpp
BINARY=gather_data_cpp.x

rm ./$BINARY
rm ./binaries/gather_data_cpp_*_*.x

for n_cells in ${N_CELLS_PER_BLOCK[@]}; do
    for n_blocks in ${N_BLOCKS[@]}; do
        # We need dx=dy for all
        echo
        echo "Building for n_cells=$n_cells / n_blocks=$n_blocks"
        echo "----------------------------------------------------------"

        # Setup constants.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataCpp/constants.h
        sed "s/N_CELLS_IN_X/$n_cells/g" \
                $TESTDIR/constants_base.h > \
                $TESTDIR/GatherDataCpp/constants.h
        sed -i "s/N_CELLS_IN_Y/$n_cells/g" $TESTDIR/GatherDataCpp/constants.h
        sed -i "s/N_CELLS_IN_Z/1/g"        $TESTDIR/GatherDataCpp/constants.h
        sed -i "s/N_DIMENSIONS/2/g"        $TESTDIR/GatherDataCpp/constants.h

        # Setup Flash.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataCpp/Flash.h
        sed "s/N_BLOCKS_ALONG_X/$n_blocks/g" \
                $TESTDIR/Flash_base.h > \
                $TESTDIR/GatherDataCpp/Flash.h
        sed -i "s/N_BLOCKS_ALONG_Y/$n_blocks/g" $TESTDIR/GatherDataCpp/Flash.h
        sed -i "s/N_BLOCKS_ALONG_Z/1/g"         $TESTDIR/GatherDataCpp/Flash.h
        sed -i "s/REFINEMENT_LEVELS/1/g"        $TESTDIR/GatherDataCpp/Flash.h

        # Build test binary
        make -f $MAKEFILE clean all
        if [[ $? -ne 0 ]]; then
            echo "Unable to compile n_cells=$n_cells / n_blocks=$n_blocks"
            exit 1;
        fi

        echo
        ldd $BINARY
        mv ./$BINARY ./binaries/gather_data_cpp_${n_blocks}_${n_cells}.x
    done
done

rm $TESTDIR/GatherDataCpp/Flash.h
rm $TESTDIR/GatherDataCpp/constants.h

make -f $MAKEFILE clean

echo
ls -lah ./binaries/gather_data_cpp_*_*.x
