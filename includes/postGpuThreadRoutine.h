#ifndef POST_GPU_THREAD_ROUTINE_H__
#define POST_GPU_THREAD_ROUTINE_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void postGpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif

