#include "Grid.h"
#include "cudaTestConstants.h"
#include "CudaStreamManager.h"

#include <gtest/gtest.h>

namespace cudaTestConstants {
    unsigned int N_WAIT_CYCLES = 0;
};

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    cudaTestConstants::N_WAIT_CYCLES = std::stoi(std::string(argv[1]));

    // Initialize Grid unit/AMReX
    Grid*    grid = Grid::instance();

    CudaStreamManager::setMaxNumberStreams(cudaTestConstants::N_STREAMS);

    int  rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int  errorCode = RUN_ALL_TESTS();

    // Finalize Grid unit/AMReX
    delete grid;
    grid = nullptr;

    return errorCode;
}

