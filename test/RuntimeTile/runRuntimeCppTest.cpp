#include "Grid.h"
#include "OrchestrationRuntime.h"

#include <gtest/gtest.h>

static constexpr unsigned int   N_TILE_THREAD_TEAMS   = 3;
static constexpr unsigned int   N_PACKET_THREAD_TEAMS = 0;
static constexpr unsigned int   MAX_THREADS           = 5;

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Instantiate Grid unit, which initializes AMReX and MPI
    orchestration::Grid::instance();

    orchestration::Runtime::setNumberThreadTeams(N_TILE_THREAD_TEAMS,
                                                 N_PACKET_THREAD_TEAMS);
    orchestration::Runtime::setMaxThreadsPerTeam(MAX_THREADS);
    orchestration::Runtime::instance();

    int  rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // When Grid is destroyed at the end of main, it triggers
    // the finalization of AMReX/MPI

    return RUN_ALL_TESTS();
}

