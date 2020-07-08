#include <AMReX.H>
#include "Grid.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        Grid::instance();

        return RUN_ALL_TESTS();
}

