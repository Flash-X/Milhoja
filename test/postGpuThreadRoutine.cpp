#include "postGpuThreadRoutine.h"

#include <cstdio>
#include <unistd.h>

#include "Block.h"

void ThreadRoutines::postGpu(const unsigned int tId,
                             const std::string& name,
                             Block& block) {
    printf("[%s / Thread %d] Post-GPU thread got block %d\n",
           name.c_str(), tId, block.index());
    usleep(250000);
}

