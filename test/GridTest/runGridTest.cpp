#include "Grid.h"
#include "OrchestrationLogger.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        orchestration::Logger::instantiate("GridUnitTest.log");

        orchestration::Grid::instantiate();

        return RUN_ALL_TESTS();
}

