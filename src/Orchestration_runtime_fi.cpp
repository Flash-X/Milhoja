#include <string>
#include <stdexcept>

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
            OrchestrationRuntime<int>::setNumberThreadTeams(static_cast<unsigned int>(nTeams));
            OrchestrationRuntime<int>::setMaxThreadsPerTeam(static_cast<unsigned int>(nThreadsPerTeam));
            OrchestrationRuntime<int>::setLogFilename(logFilename);

            OrchestrationRuntime<int>*  runtime = OrchestrationRuntime<int>::instance();
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
    void   orchestration_execute_tasks_fi(TASK_FCN<int> cpuTask) {
        int tId  = 2;
        int work = 12;
        cpuTask(tId, work); 
    }

    /**
     *
     */
    void   orchestration_finalize_fi(void) {
        delete OrchestrationRuntime<int>::instance();
    }
}

