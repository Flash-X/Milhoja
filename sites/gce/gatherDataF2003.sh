#!/bin/bash

######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildGatherDataF2003.sh
. $FLASH_INTEL_SETUP

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
module list

date
for binary in ./binaries/gather_data_F2003_*_*.x; do
    echo
    echo $binary
    echo
    ldd $binary
    echo

    time $binary
done
date

