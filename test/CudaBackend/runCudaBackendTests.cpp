#include "CudaStreamManager.h"
#include "OrchestrationLogger.h"

#include <gtest/gtest.h>

namespace cudaTestConstants {
    unsigned int SLEEP_TIME_NS = 0;
};

int main(int argc, char* argv[]) {
    constexpr int   N_STREAMS = 3;

    ::testing::InitGoogleTest(&argc, argv);

    orchestration::Logger::setLogFilename("CudaBackend.log");

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    cudaTestConstants::SLEEP_TIME_NS = std::stoi(std::string(argv[1]));

    // Instantiate up front so that the acquisition of stream resources is not
    // included in the timing of the first test.
    orchestration::CudaStreamManager::setMaxNumberStreams(N_STREAMS);
    orchestration::CudaStreamManager::instance();

    return RUN_ALL_TESTS();
}

