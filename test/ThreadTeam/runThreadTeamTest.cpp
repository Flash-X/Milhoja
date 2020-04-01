#include <mpi.h>
#include <gtest/gtest.h>

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

    // This test does not need MPI.  However, I want to test on multiple
    // socket systems but with the test process pinned to a single socket.
    // The present hack is to configure a job with 1 MPI rank/socket
    // and suppress the output from the second rank.
    MPI_Init(&argc, &argv);
    int  rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int   errorCode = RUN_ALL_TESTS();
    MPI_Finalize();

    return errorCode;
}

