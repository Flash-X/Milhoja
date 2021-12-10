#include "OrchestrationLogger.h"

#include <gtest/gtest.h>

#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    orchestration::Logger::instantiate("RuntimeTest.log",
                                       GLOBAL_COMM, LEAD_RANK);

    return RUN_ALL_TESTS();
}

