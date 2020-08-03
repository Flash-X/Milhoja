#!/bin/bash

######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildTestRuntimePacketCpp.sh
module purge
module load intel/19.0
module load mpich/3.3.2-intel
module list

time ./binaries/test_runtime_packet_cpp.x 

