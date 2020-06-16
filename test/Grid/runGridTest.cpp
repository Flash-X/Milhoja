#include <AMReX.H>
#include "Grid.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        Grid::instance();

        //int rank = -1;
        //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        return RUN_ALL_TESTS();
}

