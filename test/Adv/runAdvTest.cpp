#include "Grid.h"
#include "OrchestrationRuntime.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        orchestration::Runtime::setLogFilename("AdvectionTest.log");

        orchestration::Grid::instantiate();

        return RUN_ALL_TESTS();
}

