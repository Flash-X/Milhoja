#include <string>
#include <stdexcept>

#include "Block.h"
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
            OrchestrationRuntime<Block>::setNumberThreadTeams(static_cast<unsigned int>(nTeams));
            OrchestrationRuntime<Block>::setMaxThreadsPerTeam(static_cast<unsigned int>(nThreadsPerTeam));
            OrchestrationRuntime<Block>::setLogFilename(logFilename);

            OrchestrationRuntime<Block>*  runtime = OrchestrationRuntime<Block>::instance();
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
    void   orchestration_finalize_fi(void) {
        delete OrchestrationRuntime<Block>::instance();
    }
}

