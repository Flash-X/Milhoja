#include <mpi.h>
#include <gtest/gtest.h>

#include "OrchestrationLogger.h"

#include "threadTeamTest.h"

namespace T3 {
    unsigned int nThreadsPerTeam = 0;
};

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    T3::nThreadsPerTeam = std::stoi(std::string(argv[1]));

    MPI_Comm   MILHOJA_MPI_COMM = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    // Each test sets its own meaningful filename
    orchestration::Logger::instantiate(MILHOJA_MPI_COMM, "DeleteMe.log");

    int   errCode = RUN_ALL_TESTS();

    MPI_Finalize();

    return errCode;
}

