#include "Grid.h"
#include "cudaTestConstants.h"
#include "CudaStreamManager.h"
#include "CudaRuntime.h"

#include <gtest/gtest.h>

namespace cudaTestConstants {
    unsigned int N_WAIT_CYCLES = 0;
};

static constexpr unsigned int   N_THREAD_TEAMS   = 1;
static constexpr unsigned int   MAX_THREADS      = 5;

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    cudaTestConstants::N_WAIT_CYCLES = std::stoi(std::string(argv[1]));

    orchestration::CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    orchestration::CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    orchestration::CudaRuntime::setLogFilename("DeleteMe.log");
    orchestration::CudaRuntime::instance().printGpuInformation();

    // Initialize Grid unit/AMReX
    orchestration::Grid::instantiate();

    orchestration::CudaStreamManager::setMaxNumberStreams(cudaTestConstants::N_STREAMS);

    int  rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Grid/AMReX/MPI will finalize when grid goes out of scope and is destroyed
    return RUN_ALL_TESTS();
}

