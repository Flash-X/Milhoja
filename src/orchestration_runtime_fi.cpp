#include <string>
#include <iostream>
#include <stdexcept>

#include "Tile.h"

#include "actionRoutine.h"
#include "ActionBundle.h"
#include "ThreadTeamDataType.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

using namespace orchestration;

extern "C" {
    /**
     * 
     */
    int    orchestration_init_fi(const int nTeams,
                                 const int nThreadsPerTeam,
                                 const int nStreams,
                                 const long long nBytesInMemoryPools,
                                 char* logFilename) {
        constexpr   std::size_t   MAX_SIZE_T = static_cast<std::size_t>(-1);

        if (nTeams < 0) {
            std::cerr << "[orchestration_init_fi] nTeams must be non-negative\n\n";
            return 0;
        } else if (nThreadsPerTeam < 0) {
            std::cerr << "[orchestration_init_fi] nThreadsPerTeam must be non-negative\n\n";
            return 0;
        } else if (nStreams < 0) {
            std::cerr << "[orchestration_init_fi] nStreams must be non-negative\n\n";
            return 0;
        } else if (nBytesInMemoryPools < 0) {
            std::cerr << "[orchestration_init_fi] nBytesInMemoryPools must be non-negative\n\n";
            return 0;
        } else if (nBytesInMemoryPools > MAX_SIZE_T) {
            std::cerr << "[orchestration_init_fi] nBytesInMemoryPools is too large\n\n";
            return 0;
        }

        try {
            orchestration::Logger::instantiate(logFilename);

            orchestration::Runtime::instantiate(static_cast<unsigned int>(nTeams),
                                                static_cast<unsigned int>(nThreadsPerTeam),
                                                static_cast<unsigned int>(nStreams),
                                                static_cast<std::size_t>(nBytesInMemoryPools));
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
    int    orchestration_execute_tasks_fi(ACTION_ROUTINE cpuAction,
                                          const int nCpuThreads,
                                          ACTION_ROUTINE gpuAction,
                                          const int nGpuThreads,
                                          ACTION_ROUTINE postGpuAction,
                                          const int nPostGpuThreads) {
        ActionBundle    bundle;
        bundle.name                          = "Action Bundle from Fortran";
        bundle.cpuAction.name                = "cpuAction";
        bundle.cpuAction.nInitialThreads     = static_cast<unsigned int>(nCpuThreads);
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = cpuAction;
        bundle.gpuAction.name                = "gpuAction";
        bundle.gpuAction.nInitialThreads     = static_cast<unsigned int>(nGpuThreads);
        bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.gpuAction.routine             = gpuAction;
        bundle.postGpuAction.name            = "postGpuAction";
        bundle.postGpuAction.nInitialThreads = static_cast<unsigned int>(nPostGpuThreads);
        bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
        bundle.postGpuAction.routine         = postGpuAction;

        try {
            orchestration::Runtime::instance().executeTasks(bundle);
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
    void   orchestration_finalize_fi(void) {}
}

