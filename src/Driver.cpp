#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <pthread.h>

#include "cpuThreadRoutine.h"
#include "gpuThreadRoutine.h"
#include "postGpuThreadRoutine.h"
#include "OrchestrationRuntime.h"

const unsigned int MAX_THREADS = 10;

int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("\nNo command line arguments please\n\n");
        exit(-1);
    }

    std::vector<int>   work = {-5, -4, -1, 0, 6, 25};

    try {
        OrchestrationRuntime    runtime(MAX_THREADS, MAX_THREADS, MAX_THREADS);

        // The work vector is a standin for the set of parameters we need to
        // specify the tile iterator to use.
        runtime.executeTask(work,
                            ThreadRoutines::cpu, 2,
                            ThreadRoutines::gpu, 5,
                            ThreadRoutines::postGpu, 0);
        printf("Runtime finished\n");
    } catch (std::logic_error  e) {
        printf("\nLOGIC ERROR: %s\n\n", e.what());
        return -2;
    } catch (std::runtime_error  e) {
        printf("\nRUNTIME ERROR: %s\n\n", e.what());
        return -3;
    } catch (...) {
        printf("\n??? ERROR: Unanticipated error\n\n");
        return -4;
    }

    pthread_exit(NULL);

    return 0;
}

