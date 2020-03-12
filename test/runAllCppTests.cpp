#include <AMReX.H>

#include "Grid.h"
#include "constants.h"

#include <gtest/gtest.h>

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Grid unit/AMReX
    Grid<NXB,NYB,NZB,NGUARD>*    grid = Grid<NXB,NYB,NZB,NGUARD>::instance();

    int  errorCode = RUN_ALL_TESTS();

    // Finalize Grid unit/AMReX
    delete grid;
    grid = nullptr;

    return errorCode;
}

