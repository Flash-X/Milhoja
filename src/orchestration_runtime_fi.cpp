#include <string>
#include <stdexcept>

#include "Tile.h"
#include "runtimeTask.h"
#include "OrchestrationRuntime.h"

extern "C" {
    /**
     * \todo - Need this to return a bool so that the client code
     *         can abort on failure
     */
    void   orchestration_init_fi(const int nTeams,
                                 const int nThreadsPerTeam,
                                 char* logFilename) {
        try {
            OrchestrationRuntime<Tile>::setNumberThreadTeams(static_cast<unsigned int>(nTeams));
            OrchestrationRuntime<Tile>::setMaxThreadsPerTeam(static_cast<unsigned int>(nThreadsPerTeam));
            OrchestrationRuntime<Tile>::setLogFilename(logFilename);

            OrchestrationRuntime<Tile>::instance();
        } catch (std::invalid_argument  e) {
            printf("\nINVALID ARGUMENT: %s\n\n", e.what());
        } catch (std::logic_error  e) {
            printf("\nLOGIC ERROR: %s\n\n", e.what());
        } catch (std::runtime_error  e) {
            printf("\nRUNTIME ERROR: %s\n\n", e.what());
        } catch (...) {
            printf("\n??? ERROR: Unanticipated error\n\n");
        }
    }

    /**
     *
     */
    void   orchestration_execute_tasks_fi(TASK_FCN<Tile> cpuTask,
                                          const int nCpuThreads,
                                          TASK_FCN<Tile> gpuTask,
                                          const int nGpuThreads,
                                          TASK_FCN<Tile> postGpuTask,
                                          const int nPostGpuThreads) {
        OrchestrationRuntime<Tile>*  runtime = OrchestrationRuntime<Tile>::instance();
        runtime->executeTask("Task1",
                             cpuTask,
                             static_cast<unsigned int>(nCpuThreads), "CpuTask",
                             gpuTask,
                             static_cast<unsigned int>(nGpuThreads), "GpuTask",
                             postGpuTask,
                             static_cast<unsigned int>(nPostGpuThreads), "postGpuTask");
    }

    /**
     *
     */
    void   orchestration_finalize_fi(void) {
        delete OrchestrationRuntime<Tile>::instance();
    }
}

