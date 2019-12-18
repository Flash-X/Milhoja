#include "postGpuThreadRoutine.h"

#include <cstdio>
#include <unistd.h>

void ThreadRoutines::postGpu(const unsigned int tId,
                             const std::string& name,
                             const int work) {
    printf("[%s %d] Post-GPU thread got work %d\n", name.c_str(), tId, work);
    usleep(250000);
}

