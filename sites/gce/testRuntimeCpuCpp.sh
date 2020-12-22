#!/bin/bash

######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildTestRuntimeCpuCpp.sh
. $FLASH_INTEL_SETUP

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
module list

time ./binaries/test_runtime_cpu_cpp.x
