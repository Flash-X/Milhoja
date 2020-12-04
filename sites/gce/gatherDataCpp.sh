#!/bin/bash

######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildGatherDataCpp.sh
. $FLASH_INTEL_SETUP

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
module list

date
for binary in ./binaries/gather_data_cpp_*_*.x; do
    time $binary
done
date

