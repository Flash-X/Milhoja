#include "cpuThreadRoutine.h"

#include <cstdio>

#include <unistd.h>

void ThreadRoutines::cpu(const unsigned int tId,
                         const std::string& name,
                         const int work) {
    printf("[%s / Thread %d] CPU thread got work %d\n", name.c_str(), tId, work);
    usleep(500000);
}

