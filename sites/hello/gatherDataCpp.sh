#!/bin/bash

# Define test problems
N_CELLS_PER_BLOCK=(8 16 32 64 128)
N_BLOCKS=(2 4 8 16 32 64)

MAKEFILE=Makefile_gatherData_cpp
BINARY=gather_data_cpp.x

TESTDIR=../../test

N_THREADS=4

rm gatherDataCpp_*_*_*_*_*_*.dat

for n_cells in ${N_CELLS_PER_BLOCK[@]}; do
    for n_blocks in ${N_BLOCKS[@]}; do
        # We need dx=dy for all

        # Setup constants.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataCpp/constants.h
        sed "s/N_CELLS_IN_X/$n_cells/g" \
                $TESTDIR/constants_base.h > \
                $TESTDIR/GatherDataCpp/constants.h
        sed -i '' "s/N_CELLS_IN_Y/$n_cells/g" $TESTDIR/GatherDataCpp/constants.h
        sed -i '' "s/N_CELLS_IN_Z/1/g"        $TESTDIR/GatherDataCpp/constants.h
        sed -i '' "s/N_DIMENSIONS/2/g"        $TESTDIR/GatherDataCpp/constants.h

        # Setup Flash.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataCpp/Flash_par.h
        sed "s/SED_REPLACE_N_BLOCKS_X/$n_blocks/g" \
                $TESTDIR/GatherDataCpp/flash.par > \
                $TESTDIR/GatherDataCpp/Flash_par.h
        sed -i '' "s/SED_REPLACE_N_BLOCKS_Y/$n_blocks/g" $TESTDIR/GatherDataCpp/Flash_par.h
        sed -i '' "s/SED_REPLACE_N_BLOCKS_Z/1/g"         $TESTDIR/GatherDataCpp/Flash_par.h
        sed -i '' "s/REFINEMENT_LEVELS/1/g"              $TESTDIR/GatherDataCpp/Flash_par.h

        # Build test binary
        make -f $MAKEFILE clean all
        if [[ $? -ne 0 ]]; then
            echo "Unable to compile n_cells=$n_cells / n_blocks=$n_blocks"
            exit 1;
        fi

        time ./$BINARY $N_THREADS
        if [[ $? -ne 0 ]]; then
            echo "Unable to execute $BINARY successfully"
            exit 2;
        fi
    done
done

rm $TESTDIR/GatherDataCpp/Flash_par.h
rm $TESTDIR/GatherDataCpp/constants.h

