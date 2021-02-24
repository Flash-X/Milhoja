#include "Grid.h"
#include "OrchestrationLogger.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        // Try instance before instantiate
        try {
            orchestration::Logger::instance();
        } catch (const std::logic_error& e) {
        }
        try {
            orchestration::Grid::instance();
        } catch (const std::logic_error& e) {
        }

        orchestration::Logger::instantiate("GridUnitTest.log");

        orchestration::Grid::instantiate();

        return RUN_ALL_TESTS();
}

