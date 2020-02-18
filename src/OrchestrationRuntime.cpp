#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

#include "Block.h"
#include "BlockIterator.h"

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
#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Initializing\n";
#endif

    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i, logFilename_);
    }

#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Initialized\n";
#endif
}

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime::~OrchestrationRuntime(void) {
#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Finalizing\n";
#endif

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }

    delete [] teams_;
    teams_ = nullptr;

    instance_ = nullptr;

#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Destroyed\n";
#endif
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTask(Grid& myGrid,
                                       const std::string& bundleName,
                                       TASK_FCN* cpuTask,
                                       const unsigned int nCpuThreads,
                                       const std::string& cpuTaskName,
                                       TASK_FCN* gpuTask, 
                                       const unsigned int nGpuThreads,
                                       const std::string& gpuTaskName, 
                                       TASK_FCN* postGpuTask,
                                       const unsigned int nPostGpuThreads,
                                       const std::string& postGpuTaskName) {
    // TODO: The pipeline construction would be done dynamically.
    // Realistically, we would have multiple different implementations and 
    // this routine would select the setup based on the parameter values.
    // The assignment of team type of team ID would be hardcoded in each.
    //
    // TASK_COMPOSER: Should the task composer identify the pipelines that it
    // will need and then write this routine for each?  If not, the
    // combinatorics could grow out of control fairly quickly.

    ThreadTeam*   cpuTeam     = teams_[0];
    ThreadTeam*   gpuTeam     = teams_[1];
    ThreadTeam*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads = nCpuThreads + nGpuThreads + nPostGpuThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTask] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }
    // TODO: Confirm that no team has publisher's/subscribers.

#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Start execution of " 
              << bundleName << std::endl;
#endif

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachWorkReceiver(postGpuTeam);

    cpuTeam->startTask(cpuTask, nCpuThreads,
                       "CpuTask", cpuTaskName);
    gpuTeam->startTask(gpuTask, nGpuThreads,
                       "GpuTask", gpuTaskName);
    postGpuTeam->startTask(postGpuTask, nPostGpuThreads,
                           "PostGpuTask", postGpuTaskName);


    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    Block         block;
    BlockIterator itor(&myGrid);
    for (itor.clear(); itor.isValid(); itor.next()) {
        block = itor.currentBlock();

        // Queue the GPU work first so that it can potentially get a head start
//        gpuTeam->enqueue(block);
//        cpuTeam->enqueue(block);
    }
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

#ifdef VERBOSE
    std::cout << "[OrchestrationRuntime] Finished execution of " 
              << bundleName << std::endl;
#endif
}

