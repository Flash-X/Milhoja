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

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        // Each test sets its own meaningful filename
        milhoja::Logger::initialize("DeleteMe.log", GLOBAL_COMM, LEAD_RANK);

        exitCode = RUN_ALL_TESTS();

        milhoja::Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - ThreadTeam::main - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - ThreadTeam::main - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

