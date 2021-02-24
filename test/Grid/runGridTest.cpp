#include "Grid.h"
#include "OrchestrationLogger.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        orchestration::Logger::instantiate("GridUnitTest.log");

        // Try Grid::instance before Grid::instantiate
        try {
            orchestration::Grid::instance();
        } catch (const std::logic_error& e) {
        }
        orchestration::Grid::instantiate();

        return RUN_ALL_TESTS();
}

