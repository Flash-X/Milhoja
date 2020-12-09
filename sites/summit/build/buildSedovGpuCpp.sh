#!/bin/bash

######################################################################
#####-----               FULLY SPECIFY TEST RUN             -----#####
######################################################################
# Specified relative to location of Makefile
PAR_FILE=sedov_2D_cartesian_gpu.par

# When running a binary created with this script, the job script
# sedovGpuCpp_lsf must be setup to use this same module setup.
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
MAKEFILE=Makefile_sedov_gpu_cpp
BINARY=sedov_gpu_cpp.x
DEBUG_BINARY=sedov_gpu_cpp_debug.x

# Setup par file for compilation
cp $PAR_FILE $SIMDIR/Flash_par.h

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

