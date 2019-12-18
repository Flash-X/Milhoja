#include "gpuThreadRoutine.h"

#include <cstdio>

#include <unistd.h>

void ThreadRoutines::gpu(const unsigned int tId,
                         const std::string& name,
                         const int work) {
    printf("[%s %d] GPU thread got work %d\n", name.c_str(), tId, work);
    usleep(100000);
}

