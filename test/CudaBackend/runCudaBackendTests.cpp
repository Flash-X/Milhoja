#include "StreamManager.h"
#include "OrchestrationLogger.h"

#include <gtest/gtest.h>

namespace cudaTestConstants {
    unsigned int SLEEP_TIME_NS = 0;
};

int main(int argc, char* argv[]) {
    // This value cannot be changed without breaking tests.
    constexpr int   N_STREAMS = 3;

    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    cudaTestConstants::SLEEP_TIME_NS = std::stoi(std::string(argv[1]));

    // Instantiate up front so that the acquisition of stream resources is not
    // included in the timing of the first test.
    orchestration::Logger::instantiate("CudaBackend.log");
    orchestration::StreamManager::instantiate(N_STREAMS);

    return RUN_ALL_TESTS();
}

