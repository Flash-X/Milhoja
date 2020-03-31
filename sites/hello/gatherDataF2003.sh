#!/bin/bash

# Define test problems
N_CELLS_PER_BLOCK=(1 2 4 8 16 32 64 128 256 512)
N_BLOCKS=(16)

MAKEFILE=Makefile_gatherData_F2003
BINARY=gather_data_F2003.x

rm gatherDataF2003_*_*_*_*_*_*.dat

for n_cells in ${N_CELLS_PER_BLOCK[@]}; do
    for n_blocks in ${N_BLOCKS[@]}; do
        # We need dx=dy for all

        # Setup constants.h with current simulation's Grid parameters
        rm ./test/constants.h
        sed "s/N_CELLS_IN_X/$n_cells/g" \
                ./test/constants_base.h > \
                ./test/constants.h
        sed -i '' "s/N_CELLS_IN_Y/$n_cells/g" ./test/constants.h
        sed -i '' "s/N_CELLS_IN_Z/1/g" ./test/constants.h

        # Setup Flash.h with current simulation's Grid parameters
        rm ./test/Flash.h
        sed "s/N_BLOCKS_ALONG_X/$n_blocks/g" \
                ./test/Flash_base.h > \
                ./test/Flash.h
        sed -i '' "s/N_BLOCKS_ALONG_Y/$n_blocks/g" ./test/Flash.h
        sed -i '' "s/N_BLOCKS_ALONG_Z/1/g" ./test/Flash.h

        # Build test binary
        make -f $MAKEFILE clean all
        if [[ $? -ne 0 ]]; then
            echo "Unable to compile n_cells=$n_cells / n_blocks=$n_blocks"
            exit 3;
        fi
        rm ./test/Flash.h
        rm ./test/constants.h

        time ./$BINARY
        if [[ $? -ne 0 ]]; then
            echo "Unable to execute $BINARY successfully"
            exit 4;
        fi
    done
done

