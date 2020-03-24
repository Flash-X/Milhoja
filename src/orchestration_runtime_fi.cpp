#include <string>
#include <iostream>
#include <stdexcept>

#include "Tile.h"
#include "runtimeTask.h"
#include "OrchestrationRuntime.h"

extern "C" {
    /**
     * 
     */
    int    orchestration_init_fi(const int nTeams,
                                 const int nThreadsPerTeam,
                                 char* logFilename) {
        try {
            OrchestrationRuntime::setNumberThreadTeams(static_cast<unsigned int>(nTeams));
            OrchestrationRuntime::setMaxThreadsPerTeam(static_cast<unsigned int>(nThreadsPerTeam));
            OrchestrationRuntime::setLogFilename(logFilename);

            OrchestrationRuntime::instance();
        } catch (std::invalid_argument& e) {
            std::cerr << "\nINVALID ARGUMENT: " << e.what() << "\n\n";
            return 0;
        } catch (std::logic_error& e) {
            std::cerr << "\nLOGIC ERROR: " << e.what() << "\n\n";
            return 0;
        } catch (std::runtime_error& e) {
            std::cerr << "\nRUNTIME ERROR: " << e.what() << "\n\n";
            return 0;
        } catch (...) {
            std::cerr << "\n??? ERROR: Unanticipated error\n\n";
            return 0;
        }

        return 1;
    }

    /**
     *
     */
    int    orchestration_execute_tasks_fi(TASK_FCN<Tile> cpuTask,
                                          const int nCpuThreads,
                                          TASK_FCN<Tile> gpuTask,
                                          const int nGpuThreads,
                                          TASK_FCN<Tile> postGpuTask,
                                          const int nPostGpuThreads) {
        try {
            OrchestrationRuntime*  runtime = OrchestrationRuntime::instance();
            runtime->executeTasks("Task1",
                                  cpuTask,
                                  static_cast<unsigned int>(nCpuThreads), "CpuTask",
                                  gpuTask,
                                  static_cast<unsigned int>(nGpuThreads), "GpuTask",
                                  postGpuTask,
                                  static_cast<unsigned int>(nPostGpuThreads), "postGpuTask");
        } catch (std::invalid_argument& e) {
            std::cerr << "\nINVALID ARGUMENT: " << e.what() << "\n\n";
            return 0;
        } catch (std::logic_error& e) {
            std::cerr << "\nLOGIC ERROR: " << e.what() << "\n\n";
            return 0;
        } catch (std::runtime_error& e) {
            std::cerr << "\nRUNTIME ERROR: " << e.what() << "\n\n";
            return 0;
        } catch (...) {
            std::cerr << "\n??? ERROR: Unanticipated error\n\n";
            return 0;
        }

        return 1;
    }

    /**
     *
     */
    void   orchestration_finalize_fi(void) {
        delete OrchestrationRuntime::instance();
    }
}

