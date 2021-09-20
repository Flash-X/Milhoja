#include "OrchestrationLogger.h"

#include <mpi.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Comm    MILHOJA_MPI_COMM = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    orchestration::Logger::instantiate(MILHOJA_MPI_COMM,
                                       "RuntimeTest.log");

    int    errCode = RUN_ALL_TESTS();

    MPI_Finalize();

    return errCode;
}

