#include "OrchestrationLogger.h"

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    orchestration::Logger::instantiate("RuntimeTest.log");

    return RUN_ALL_TESTS();
}

