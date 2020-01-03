#include "cpuThreadRoutine.h"

#include <cstdio>
#include <unistd.h>

#include "Block.h"

void ThreadRoutines::cpu(const unsigned int tId,
                         const std::string& name,
                         Block& block) {
    printf("[%s / Thread %d] CPU thread got block %d\n",
           name.c_str(), tId, block.index());
    usleep(500000);
}

