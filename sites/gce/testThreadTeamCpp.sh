#!/bin/bash
 
######################################################################
#####-----              FULLY SPECIFY TEST RUN              -----#####
######################################################################
N_THREADS_PER_TEAM=16

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildTestThreadTeamCpp.sh
module purge
module load intel/19.0
module list

time ./binaries/test_threadteam_cpp.x $N_THREADS_PER_TEAM

