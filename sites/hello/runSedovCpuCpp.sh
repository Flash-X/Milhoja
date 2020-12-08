#!/bin/bash

MAKEFILE=Makefile_sedov_cpu_cpp
BINARY=sedov_cpu_cpp.x
N_PROCS=2

SIMDIR=../../simulations/Sedov

cp ./sedov_2D_cartesian_cpu.par $SIMDIR/Flash_par.h

# Build test binary
make -f $MAKEFILE clean all

time mpirun -np $N_PROCS ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 1;
fi

rm $SIMDIR/Flash_par.h

