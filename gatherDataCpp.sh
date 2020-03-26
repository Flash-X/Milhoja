#!/bin/bash

N_CELLS_PER_BLOCK=(8 16 32 64 128)
N_BLOCKS=(2 4 8 16 32 64)

rm gatherData_*_*_*_*_*_*_cpp.dat

for n_cells in ${N_CELLS_PER_BLOCK[@]}; do
    for n_blocks in ${N_BLOCKS[@]}; do
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

        make -f Makefile_gatherData_cpp clean all
        if [[ $? -ne 0 ]]; then
            echo "Unable to compile n_cells=$n_cells / n_blocks=$n_blocks"
            exit 1;
        fi
        rm ./test/Flash.h
        rm ./test/constants.h

        ./gather_data_cpp.x
        if [[ $? -ne 0 ]]; then
            echo "Unable to execute gather_data_cpp.x successfully"
            exit 2;
        fi
    done
done

