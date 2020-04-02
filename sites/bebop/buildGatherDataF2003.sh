#!/bin/bash

# Define test problems
N_CELLS_PER_BLOCK=(1 2 4 8 16 32 64 128 256 512)
N_BLOCKS=(16)

MAKEFILE=Makefile_gatherData_F2003
BINARY=gather_data_F2003.x

TESTDIR=../../test

rm ./$BINARY
rm ./binaries/gather_data_F2003_*_*.x

module purge
module load git
module load gcc/7.1.0-4bgguyp
module load mpich/3.2-bsq4vhr
module list

for n_cells in ${N_CELLS_PER_BLOCK[@]}; do
    for n_blocks in ${N_BLOCKS[@]}; do
        # We need dx=dy for all

        # Setup constants.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataF2003/constants.h
        sed "s/N_CELLS_IN_X/$n_cells/g" \
                $TESTDIR/constants_base.h > \
                $TESTDIR/GatherDataF2003/constants.h
        sed -i "s/N_CELLS_IN_Y/$n_cells/g" $TESTDIR/GatherDataF2003/constants.h
        sed -i "s/N_CELLS_IN_Z/1/g"        $TESTDIR/GatherDataF2003/constants.h

        # Setup Flash.h with current simulation's Grid parameters
        rm $TESTDIR/GatherDataF2003/Flash.h
        sed "s/N_BLOCKS_ALONG_X/$n_blocks/g" \
                $TESTDIR/Flash_base.h > \
                $TESTDIR/GatherDataF2003/Flash.h
        sed -i "s/N_BLOCKS_ALONG_Y/$n_blocks/g" $TESTDIR/GatherDataF2003/Flash.h
        sed -i "s/N_BLOCKS_ALONG_Z/1/g"         $TESTDIR/GatherDataF2003/Flash.h

        # Build test binary
        make -f $MAKEFILE clean all
        if [[ $? -ne 0 ]]; then
            echo "Unable to compile n_cells=$n_cells / n_blocks=$n_blocks"
            exit 1;
        fi

        mv ./$BINARY ./binaries/gather_data_F2003_${n_blocks}_${n_cells}.x
    done
done

rm $TESTDIR/GatherDataF2003/Flash.h
rm $TESTDIR/GatherDataF2003/constants.h
 
make -f $MAKEFILE clean

