#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

#include "Grid.h"
#include "Flash.h"

std::string           OrchestrationRuntime::logFilename_       = "";
unsigned int          OrchestrationRuntime::nTeams_            = 1;
unsigned int          OrchestrationRuntime::maxThreadsPerTeam_ = 5;
OrchestrationRuntime* OrchestrationRuntime::instance_          = nullptr;

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime* OrchestrationRuntime::instance(void) {
    if (!instance_) {
        instance_ = new OrchestrationRuntime();
    }

    return instance_;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setLogFilename(const std::string& filename) {
    if (instance_) {
        throw std::logic_error("[OrchestrationRuntime::setLogFilename] "
                               "Set only when runtime does not exist");
    }

    logFilename_ = filename;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setNumberThreadTeams(const unsigned int nTeams) {
    if (instance_) {
        throw std::logic_error("[OrchestrationRuntime::setNumberThreadTeams] "
                               "Set only when runtime does not exist");
    } else if(nTeams == 0) {
        throw std::invalid_argument("[OrchestrationRuntime::setNumberThreadTeams] "
                                    "Need at least one ThreadTeam");
    }

    nTeams_ = nTeams;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setMaxThreadsPerTeam(const unsigned int nThreads) {
    if (instance_) {
        throw std::logic_error("[OrchestrationRuntime::setMaxThreadsPerTeam] "
                               "Set only when runtime does not exist");
    } else if (nThreads == 0) {
        throw std::invalid_argument("[OrchestrationRuntime::setMaxThreadsPerTeam] "
                                    "Need at least one thread per team");
    }

    maxThreadsPerTeam_ = nThreads;
}

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime::OrchestrationRuntime(void) {
#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Initializing\n";
    logFile_.close();
#endif

    teams_ = new ThreadTeam<Tile>*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam<Tile>(maxThreadsPerTeam_, i, logFilename_);
    }

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Initialized\n";
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime::~OrchestrationRuntime(void) {
#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finalizing\n";
    logFile_.close();
#endif

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }

    delete [] teams_;
    teams_ = nullptr;

    instance_ = nullptr;

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finalized\n";
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTasks(const ActionBundle& bundle) {
#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Start execution of " 
             << bundle.name << std::endl;
    logFile_.close();
#endif

    if      (bundle.distribution != WorkDistribution::Concurrent) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks] "
                               "The runtime only handles concurrent work distriution");
    }

    bool  hasCpuAction     = (bundle.cpuAction.routine     != nullptr);
    bool  hasGpuAction     = (bundle.gpuAction.routine     != nullptr);
    bool  hasPostGpuAction = (bundle.postGpuAction.routine != nullptr);

    // TODO: The pipeline construction would be done dynamically.
    // Realistically, we would have multiple different implementations and 
    // this routine would select the setup based on the parameter values.
    // The assignment of team type of team ID would be hardcoded in each.
    //
    // TASK_COMPOSER: Should the task composer identify the pipelines that it
    // will need and then write this routine for each?  If not, the
    // combinatorics could grow out of control fairly quickly.
    if (hasCpuAction && !hasGpuAction && !hasPostGpuAction) {
        executeCpuTasks(bundle.name,
                        bundle.cpuAction);
    } else if (hasCpuAction && hasGpuAction && !hasPostGpuAction) {
        executeConcurrentCpuGpuTasks(bundle.name, 
                                     bundle.cpuAction,
                                     bundle.gpuAction);
    } else if (hasCpuAction && hasGpuAction && hasPostGpuAction) {
        executeTasks_Full(bundle.name,
                          bundle.cpuAction,
                          bundle.gpuAction,
                          bundle.postGpuAction);
    } else {
        std::string   errMsg =   "No compatible thread team layout - ";
        errMsg += bundle.name;
        errMsg += "\n";
        throw std::logic_error(errMsg);
    }

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finished execution of " 
             << bundleName << std::endl;
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTasks_Full(const std::string& bundleName,
                                             const RuntimeAction& cpuAction,
                                             const RuntimeAction& gpuAction,
                                             const RuntimeAction& postGpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given post-GPU action should run on block-based "
                               "thread team, which is not in configuration");
    }

    ThreadTeam<Tile>*   cpuTeam     = teams_[0];
    ThreadTeam<Tile>*   gpuTeam     = teams_[1];
    ThreadTeam<Tile>*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachWorkReceiver(postGpuTeam);

    // TODO: These should take a reference to a RuntimeAction
    cpuTeam->startTask(cpuAction.routine,
                       cpuAction.nInitialThreads,
                       "Concurrent_CPU_Block_Team",
                       cpuAction.name);
    gpuTeam->startTask(gpuAction.routine,
                       gpuAction.nInitialThreads,
                       "Concurrent_GPU_Block_Team",
                       gpuAction.name);
    postGpuTeam->startTask(postGpuAction.routine,
                           postGpuAction.nInitialThreads,
                           "Post_GPU_Block_Team",
                           postGpuAction.name);

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    unsigned int   level = 0;
    Grid*   grid = Grid::instance();
    for (amrex::MFIter  itor(grid->unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        // Ownership of tile resources is transferred to last team
        cpuTeam->enqueue(work, false);
        gpuTeam->enqueue(work, true);
    }
    grid = nullptr;
    gpuTeam->closeTask();
    cpuTeam->closeTask();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    // CPU and GPU pools are not dependent on any other pools
    //   => call these first
    cpuTeam->wait();
    cpuTeam->detachThreadReceiver();

    gpuTeam->wait();
    
    // The GPU pool has no more threads or work to push to its dependents
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachWorkReceiver();

    // Post-GPU follows GPU in the thread pool work/thread pipeline
    //   => the gpuTeam wait() method *must* terminate before the postGpuTeam
    //      wait() method is called to ensure that all GPU work are queued
    //      in postGpuTeam before the post-GPU thread pool can begin
    //      determining if it should terminate.
    postGpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeConcurrentCpuGpuTasks(const std::string& bundleName,
                                                        const RuntimeAction& cpuAction,
                                                        const RuntimeAction& gpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    }

    ThreadTeam<Tile>*   cpuTeam = teams_[0];
    ThreadTeam<Tile>*   gpuTeam = teams_[1];

    cpuTeam->startTask(cpuAction.routine,
                       cpuAction.nInitialThreads,
                       "Concurrent_CPU_Block_Team",
                       cpuAction.name);
    gpuTeam->startTask(gpuAction.routine,
                       gpuAction.nInitialThreads,
                       "Concurrent_GPU_Block_Team",
                       gpuAction.name);

    unsigned int   level = 0;
    Grid*   grid = Grid::instance();
    for (amrex::MFIter  itor(grid->unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        // Ownership of tile resources is transferred to last team
        cpuTeam->enqueue(work, false);
        gpuTeam->enqueue(work, true);
    }
    grid = nullptr;
    gpuTeam->closeTask();
    cpuTeam->closeTask();

    cpuTeam->wait();
    gpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeCpuTasks(const std::string& bundleName,
                                           const RuntimeAction& cpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeCpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    }

    ThreadTeam<Tile>*   cpuTeam     = teams_[0];

    cpuTeam->startTask(cpuAction.routine,
                       cpuAction.nInitialThreads,
                       "Concurrent_CPU_Block_Team",
                       cpuAction.name);

    unsigned int   level = 0;
    Grid*   grid = Grid::instance();
    for (amrex::MFIter  itor(grid->unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        // Ownership of tile resources is transferred immediately
        cpuTeam->enqueue(work, true);
    }
    grid = nullptr;
    cpuTeam->closeTask();
    cpuTeam->wait();
}

