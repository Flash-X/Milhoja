#!/bin/bash

#####----- Task Functions used in all CPU-based tests
TF_ALL=('cpu_tf_ic' \
        'cpu_tf_dens' \
        'cpu_tf_ener' \
        'cpu_tf_fused')

#####----- Setup
# Use the tools located in the same clone as this script
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
CLONEPATH=$(realpath $SCRIPTPATH/../../..)
TOOLPATH=$CLONEPATH/tools

GEN_TF_TOOL=$TOOLPATH/generate_cpp_task_function.py
GEN_TWRAPPER_TOOL=$TOOLPATH/generate_tile_wrapper.py

#####----- A generar codigo!
for taskFunction in ${TF_ALL[@]}; do
    # Scripts will not overwrite files
    rm ${taskFunction}.h
    rm ${taskFunction}.cpp
    rm Tile_${taskFunction}.h
    rm Tile_${taskFunction}.cpp

    echo
    echo "Generate code for Task Function $taskFunction"
    echo "------------------------------------------------------------"
    $GEN_TF_TOOL ${taskFunction}.json ${taskFunction}.h ${taskFunction}.cpp
    $GEN_TWRAPPER_TOOL ${taskFunction}.json Tile_${taskFunction}.h Tile_${taskFunction}.cpp
done
echo

