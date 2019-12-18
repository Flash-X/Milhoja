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

// TASK_COMPOSER: The offline tool will need to determine how many thread teams
// are needed as well as how many threads to allocate to each.
const unsigned int N_THREAD_TEAMS = 3;
const unsigned int MAX_THREADS = 10;

int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("\nNo command line arguments please\n\n");
        return 1;
    }

    std::vector<int>   work = {-5, -4, -1, 0, 6, 25};

    try {
        OrchestrationRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
        OrchestrationRuntime::setMaxThreadsPerTeam(MAX_THREADS);
        OrchestrationRuntime*    runtime = OrchestrationRuntime::instance();

        // The work vector is a standin for the set of parameters we need to
        // specify the tile iterator to use.
        runtime->executeTask(work, "Task Bundle 1",
                             ThreadRoutines::cpu, 2, "bundle1_cpuTask",
                             ThreadRoutines::gpu, 5, "bundle1_gpuTask",
                             ThreadRoutines::postGpu, 0, "bundle1_postGpuTask");

        delete runtime;
    } catch (std::invalid_argument  e) {
        printf("\nINVALID ARGUMENT: %s\n\n", e.what());
        return 2;
    } catch (std::logic_error  e) {
        printf("\nLOGIC ERROR: %s\n\n", e.what());
        return 3;
    } catch (std::runtime_error  e) {
        printf("\nRUNTIME ERROR: %s\n\n", e.what());
        return 4;
    } catch (...) {
        printf("\n??? ERROR: Unanticipated error\n\n");
        return 5;
    }

    pthread_exit(NULL);

    return 0;
}

