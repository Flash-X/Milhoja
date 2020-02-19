#include "gpuThreadRoutine.h"

#include <cstdio>
#include <unistd.h>

#include "Block.h"

void ThreadRoutines::gpu(const unsigned int tId,
                         const std::string& name,
                         Block& block) {
    printf("[%s / Thread %d] GPU thread got work %d\n",
           name.c_str(), tId, block.index());
    usleep(100000);
}

