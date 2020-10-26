#ifndef SLEEPY_GO_BYE_BYE_H__
#define SLEEPY_GO_BYE_BYE_H__

#include "DataItem.h"

namespace ActionRoutines {
    constexpr int SLEEP_US = 1000;

    void sleepyGoByeBye_tile_cpu(const int tId, orchestration::DataItem* dataItem);

#ifdef ENABLE_CUDA_OFFLOAD
    void sleepyGoByeBye_packet_cuda_gpu(const int tId,
                                        orchestration::DataItem* dataItem);
#endif
}

#endif

