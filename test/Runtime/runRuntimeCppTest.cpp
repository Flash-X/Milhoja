#include <AMReX.H>

#include "Grid.h"
#include "Flash.h"
#include "constants.h"

#include <gtest/gtest.h>

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Grid unit/AMReX
    Grid::instance();

    int  rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Grid/AMReX/MPI will finalize when grid is automatically destroyed at the
    // end of main

    return RUN_ALL_TESTS();
}

