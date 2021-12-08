#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include <gtest/gtest.h>

#include <mpi.h>

// It appears that OpenACC on Summit with PGI has max 32 asynchronous
// queues.  If you assign more CUDA streams to queues with OpenACC, then
// these streams just roll over and the last 32 CUDA streams will be the
// only streams mapped to queues.
constexpr int            N_STREAMS = 32; 
constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   N_THREADS_PER_TEAM = 10;
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 4294967296; 

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    constexpr MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    constexpr int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // Grid initialized AMReX and MPI
    orchestration::Logger::instantiate("RuntimeTest.log",
                                       GLOBAL_COMM, LEAD_RANK);
    orchestration::Runtime::instantiate(N_THREAD_TEAMS, N_THREADS_PER_TEAM,
                                        N_STREAMS, MEMORY_POOL_SIZE_BYTES);
    orchestration::Grid::instantiate();

    int  rank = -1;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Grid/AMReX/MPI will finalize when grid goes out of scope and is destroyed
    return RUN_ALL_TESTS();
}

