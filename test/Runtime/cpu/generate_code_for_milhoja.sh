#!/bin/bash

#####----- Task Functions used in all CPU-based tests
TF_ALL=('cpu_tf_ic' \
        'cpu_tf_dens' \
        'cpu_tf_ener' \
        'cpu_tf_fused' \
        'cpu_tf_analysis')

#####----- Command Line Arguments
if [[ "$#" -ne 3 ]]; then
    echo "Must pass destination folder, Makefile, dimensionality"
    exit 1
fi

SCRIPTPATH=$( cd "$(dirname "$0")" ; pwd -P )
GENPATH=$1
if [[ -d "$GENPATH" ]]; then
  echo "$GENPATH does exist."
fi

# TODO: Confirm that Makefile doesn't already exist
MAKEFILE=$2

DIM=$3

#####----- Setup
# Use the tools located in the same clone as this script
CLONEPATH=$SCRIPTPATH/../../..

TOOLPATH=$CLONEPATH/tools
GEN_TF_TOOL=$TOOLPATH/generate_task_function.py
GEN_DATAITEM_TOOL=$TOOLPATH/generate_data_item.py
MILHOJA_JSON_FMT='Milhoja-JSON'

CGPATH=$CLONEPATH/test/Base/code_generation

#####----- Makefile boilerplate
rm -f $MAKEFILE
echo "CXXFLAGS_GENERATED_DEBUG = -I${GENPATH}" >  $MAKEFILE
echo "CXXFLAGS_GENERATED_PROD  = -I${GENPATH}" >> $MAKEFILE
echo ""                                        >> $MAKEFILE
echo "SRCS_GENERATED = \\"                     >> $MAKEFILE

#####----- A generar codigo!
for taskFunction in ${TF_ALL[@]}; do
    rm -f ${GENPATH}/${taskFunction}.h
    rm -f ${GENPATH}/${taskFunction}.cpp
    rm -f ${GENPATH}/Tile_${taskFunction}.h
    rm -f ${GENPATH}/Tile_${taskFunction}.cpp

    echo
    echo "Generate code for Task Function $taskFunction"
    echo "------------------------------------------------------------"
    $GEN_TF_TOOL ${CGPATH}/${taskFunction}.json ${MILHOJA_JSON_FMT} ${GENPATH}/${taskFunction}.h ${GENPATH}/${taskFunction}.cpp
    $GEN_DATAITEM_TOOL ${CGPATH}/${taskFunction}.json ${MILHOJA_JSON_FMT} ${GENPATH}/Tile_${taskFunction}.h ${GENPATH}/Tile_${taskFunction}.cpp

    echo -e "\t${GENPATH}/${taskFunction}.cpp \\"      >> $MAKEFILE
    echo -e "\t${GENPATH}/Tile_${taskFunction}.cpp \\" >> $MAKEFILE
done
echo

