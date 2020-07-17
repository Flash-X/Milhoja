#!/bin/bash

######################################################################
#####-----             DO NOT ALTER LINES BELOW             -----#####
######################################################################
# This should match the SW stack used to build the test binaries
# as specified in buildThreadTeamValgrind.sh
module purge
module load gcc/8.3.0-fjpc5ys
module list

valgrind --tool=memcheck --show-leak-kinds=all --leak-check=full --track-origins=yes --suppressions=ThreadTeamValgrind.supp --error-exitcode=10 ./binaries/test_threadteam_valgrind.x

