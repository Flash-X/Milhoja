#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>

#include "threadTeamTest.h"

namespace T3 {
    unsigned int nThreadsPerTeam = 0;
};

int main(int argc, char* argv[]) {
    MPI_Comm  GLOBAL_COMM = MPI_COMM_WORLD;
    int       LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    T3::nThreadsPerTeam = std::stoi(std::string(argv[1]));

    // Each test sets its own meaningful filename
    milhoja::Logger::instantiate("DeleteMe.log", GLOBAL_COMM, LEAD_RANK);

    return RUN_ALL_TESTS();
}

