#include "Grid.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        orchestration::Grid::instantiate();

        return RUN_ALL_TESTS();
}

