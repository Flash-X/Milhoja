#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Determine problem dimension
# The setting of NDIM is also used by the Makefile to choose the 
# correct Grid library
if (( "$#" != 1 )); then
    echo "Please enter the dimension of the problem to solve (2 or 3)"
    exit 1
elif (( $1 == 2 || $1 == 3 )); then
    export NDIM=$1
else
    echo "Please enter the dimension of the problem to solve (2 or 3)"
    exit 2
fi

# Specified relative to location of Makefile
PAR_FILE=sedov_${NDIM}D_cartesian_cpu.par

# When running a binary created with this script, the job script
# sedovCpuCpp_lsf must be setup to use this same module setup.
. $RUNTIME_PGI_SETUP

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
echo
date
echo
module list
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
SIMDIR=../../simulations/Sedov
MAKEFILE=Makefile_sedov_baseline_cpp
BINARY=sedov_baseline_cpp.x
DEBUG_BINARY=sedov_baseline_cpp_debug.x

# Setup par file for compilation
rm $SIMDIR/Flash_par.h
rm $SIMDIR/Flash.h
rm $SIMDIR/constants.h

cp $PAR_FILE $SIMDIR/Flash_par.h
cp $SIMDIR/Flash_${NDIM}D.h $SIMDIR/Flash.h
cp $SIMDIR/constants_${NDIM}D.h $SIMDIR/constants.h

# Build non-debug mode
echo
echo "Building production version"
echo "----------------------------------------------------------"
make -f $MAKEFILE clean all
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 4;
fi
mv $BINARY ./binaries

echo
ls -lah ./binaries/$BINARY
ldd ./binaries/$BINARY
echo

rm $SIMDIR/Flash_par.h
rm $SIMDIR/Flash.h
rm $SIMDIR/constants.h

