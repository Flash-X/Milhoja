#!/bin/bash

MAKEFILE=Makefile_sedov_cpp
BINARY=sedov_cpp.x
N_PROCS=2

# Build test binary
make -f $MAKEFILE clean all

time mpirun -np $N_PROCS ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 1;
fi

