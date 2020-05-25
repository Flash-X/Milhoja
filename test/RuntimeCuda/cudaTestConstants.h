#ifndef CUDA_TEST_CONSTANTS_H__
#define CUDA_TEST_CONSTANTS_H__

#include <cstddef>

namespace cudaTestConstants {
    constexpr std::size_t    N_STREAMS = 3;
    extern unsigned int      N_WAIT_CYCLES;
};

#endif

