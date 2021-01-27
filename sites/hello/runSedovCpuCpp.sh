#!/bin/bash

MAKEFILE=Makefile_sedov_cpu_cpp
BINARY=sedov_cpu_cpp.x
N_PROCS=2

SIMDIR=../../simulations/Sedov

if (( "$#" != 1 )); then
    echo "Please enter the dimension of the problem to solve (2 or 3)"
    exit 1
elif (( $1 == 2 || $1 == 3 )); then
    export NDIM=$1
else
    echo "Please enter the dimension of the problem to solve (2 or 3)"
    exit 2
fi

cp ./sedov_${NDIM}D_cartesian_cpu.par $SIMDIR/Flash_par.h
cp $SIMDIR/Flash_${NDIM}D.h $SIMDIR/Flash.h
cp $SIMDIR/constants_${NDIM}D.h $SIMDIR/constants.h

# Build test binary
make -f $MAKEFILE clean all

time mpirun -np $N_PROCS ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 1;
fi

rm $SIMDIR/Flash_par.h
rm $SIMDIR/Flash.h
rm $SIMDIR/constants.h

