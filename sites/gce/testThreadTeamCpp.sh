#!/bin/bash
 
######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildTestThreadTeamCpp.sh
. $FLASH_INTEL_SETUP

N_THREADS_PER_TEAM=16

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
module list

time ./binaries/test_threadteam_cpp.x $N_THREADS_PER_TEAM

