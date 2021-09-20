#include "Grid.h"
#include "OrchestrationLogger.h"

#include <mpi.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        MPI_Comm   MILHOJA_MPI_COMM = MPI_COMM_WORLD;

        MPI_Init(&argc, &argv);

        orchestration::Logger::instantiate(MILHOJA_MPI_COMM,
                                           "GridUnitTest.log");

        orchestration::Grid::instantiate(MILHOJA_MPI_COMM);

        int  errCode = RUN_ALL_TESTS();

        orchestration::Grid::instance().finalize();

        MPI_Finalize();

        return errCode;
}

