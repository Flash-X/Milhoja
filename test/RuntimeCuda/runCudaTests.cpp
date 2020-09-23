#include "Grid.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"
#include "CudaRuntime.h"

#include <gtest/gtest.h>

// It appears that OpenACC on Summit with PGI has max 32 asynchronous
// queues.  If you assign more CUDA streams to queues with OpenACC, then
// these streams just roll over and the last 32 CUDA streams will be the
// only streams mapped to queues.
constexpr int            N_STREAMS = 32; 
constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   MAX_THREADS = 7;
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 4294967296; 

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    orchestration::CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    orchestration::CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    orchestration::CudaRuntime::setLogFilename("DeleteMe.log");
    std::cout << "\n";
    std::cout << "----------------------------------------------------------\n";
    orchestration::CudaRuntime::instance().printGpuInformation();
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::endl;

    orchestration::CudaStreamManager::setMaxNumberStreams(N_STREAMS);
    orchestration::CudaStreamManager::instance();

    orchestration::CudaMemoryManager::setBufferSize(MEMORY_POOL_SIZE_BYTES);
    orchestration::CudaMemoryManager::instance();

    // Initialize Grid unit/AMReX
    orchestration::Grid::instantiate();

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

