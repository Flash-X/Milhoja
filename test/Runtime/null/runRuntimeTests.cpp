#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>

int main(int argc, char* argv[]) {
    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    milhoja::Logger::instantiate("RuntimeTest.log",
                                 GLOBAL_COMM, LEAD_RANK);

    int exitCode = RUN_ALL_TESTS();

    MPI_Finalize();

    return exitCode;
}

