#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include <mpi.h>
#include <gtest/gtest.h>

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
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Comm    MILHOJA_MPI_COMM = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    // Grid initialized AMReX and MPI
    orchestration::Logger::instantiate(MILHOJA_MPI_COMM,
                                       "RuntimeTest.log");
    orchestration::Runtime::instantiate(N_THREAD_TEAMS, N_THREADS_PER_TEAM,
                                        N_STREAMS, MEMORY_POOL_SIZE_BYTES);
    orchestration::Grid::instantiate(MILHOJA_MPI_COMM);

    int  rank = -1;
    MPI_Comm_rank(MILHOJA_MPI_COMM, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int   errCode = RUN_ALL_TESTS();

    orchestration::Grid::instance().finalize();
    MPI_Finalize();

    return errCode;
}

