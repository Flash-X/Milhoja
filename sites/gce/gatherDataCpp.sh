#!/bin/bash

######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
N_THREADS=16

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildGatherDataCpp.sh
module purge
module load intel/19.0
module load mpich/3.3.2-intel
module list

date
for binary in ./binaries/gather_data_cpp_*_*.x; do
    time $binary $N_THREADS
done
date

