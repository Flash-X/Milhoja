#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

#include "Grid.h"
#include "Flash.h"

template<typename W>
std::string              OrchestrationRuntime<W>::logFilename_       = "";
template<typename W>
unsigned int             OrchestrationRuntime<W>::nTeams_            = 1;
template<typename W>
unsigned int             OrchestrationRuntime<W>::maxThreadsPerTeam_ = 5;
template<typename W>
OrchestrationRuntime<W>* OrchestrationRuntime<W>::instance_          = nullptr;

/**
 * 
 *
 * \return 
 */
template<typename W>
OrchestrationRuntime<W>* OrchestrationRuntime<W>::instance(void) {
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
template<typename W>
void OrchestrationRuntime<W>::setLogFilename(const std::string& filename) {
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
template<typename W>
void OrchestrationRuntime<W>::setNumberThreadTeams(const unsigned int nTeams) {
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
template<typename W>
void OrchestrationRuntime<W>::setMaxThreadsPerTeam(const unsigned int nThreads) {
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
template<typename W>
OrchestrationRuntime<W>::OrchestrationRuntime(void) {
#ifdef VERBOSE
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Initializing\n";
    logFile_.close();
#endif

    teams_ = new ThreadTeam<W>*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam<W>(maxThreadsPerTeam_, i, logFilename_);
    }

#ifdef VERBOSE
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
template<typename W>
OrchestrationRuntime<W>::~OrchestrationRuntime(void) {
#ifdef VERBOSE
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

#ifdef VERBOSE
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
template<typename W>
void OrchestrationRuntime<W>::executeTasks(const std::string& bundleName,
                                           TASK_FCN<W> cpuTask,
                                           const unsigned int nCpuThreads,
                                           const std::string& cpuTaskName,
                                           TASK_FCN<W> gpuTask, 
                                           const unsigned int nGpuThreads,
                                           const std::string& gpuTaskName, 
                                           TASK_FCN<W> postGpuTask,
                                           const unsigned int nPostGpuThreads,
                                           const std::string& postGpuTaskName) {
#ifdef VERBOSE
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Start execution of " 
             << bundleName << std::endl;
    logFile_.close();
#endif

    // TODO: The pipeline construction would be done dynamically.
    // Realistically, we would have multiple different implementations and 
    // this routine would select the setup based on the parameter values.
    // The assignment of team type of team ID would be hardcoded in each.
    //
    // TASK_COMPOSER: Should the task composer identify the pipelines that it
    // will need and then write this routine for each?  If not, the
    // combinatorics could grow out of control fairly quickly.
    if (cpuTask && !gpuTask && !postGpuTask) {
        executeCpuTask(bundleName,
                       cpuTask, nCpuThreads, cpuTaskName);
    } else if (cpuTask && gpuTask && postGpuTask) {
        executeTasks_Full(bundleName,
                          cpuTask,     nCpuThreads,     cpuTaskName,
                          gpuTask,     nGpuThreads,     gpuTaskName,
                          postGpuTask, nPostGpuThreads, postGpuTaskName);
    } else {
        // TODO: Return an error code so that Fortran wrapper can call
        // Driver_abortFlash with error message generated here?
        std::string   errMsg =   "No compatible thread team layout - ";
        errMsg += bundleName;
        errMsg += "\n";
        throw std::logic_error(errMsg);
    }

#ifdef VERBOSE
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
template<typename W>
void OrchestrationRuntime<W>::executeTasks_Full(const std::string& bundleName,
                                                TASK_FCN<W> cpuTask,
                                                const unsigned int nCpuThreads,
                                                const std::string& cpuTaskName,
                                                TASK_FCN<W> gpuTask, 
                                                const unsigned int nGpuThreads,
                                                const std::string& gpuTaskName, 
                                                TASK_FCN<W> postGpuTask,
                                                const unsigned int nPostGpuThreads,
                                                const std::string& postGpuTaskName) {
    ThreadTeam<W>*   cpuTeam     = teams_[0];
    ThreadTeam<W>*   gpuTeam     = teams_[1];
    ThreadTeam<W>*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads = nCpuThreads + nGpuThreads + nPostGpuThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTask] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }
    // TODO: Confirm that no team has publisher's/subscribers.


    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachWorkReceiver(postGpuTeam);

    cpuTeam->startTask(cpuTask, nCpuThreads, "CpuTask", cpuTaskName);
    gpuTeam->startTask(gpuTask, nGpuThreads, "GpuTask", gpuTaskName);
    postGpuTeam->startTask(postGpuTask, nPostGpuThreads, "PostGpuTask", postGpuTaskName);

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    // TODO: This is where it is clear that this class does not need to be a
    // template.  This iterator isn't useful if the unit of work is the int.
    // The runtime will have to cater to all the units of work across all
    // ThreadTeams and queue appropriately.
    Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
    for (amrex::MFIter  itor(grid->unk()); itor.isValid(); ++itor) {
        W  work(itor);
        cpuTeam->enqueue(work);
        gpuTeam->enqueue(work);
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
template<typename W>
void OrchestrationRuntime<W>::executeCpuTask(const std::string& bundleName,
                                             TASK_FCN<W> cpuTask,
                                             const unsigned int nCpuThreads,
                                             const std::string& cpuTaskName) {
    ThreadTeam<W>*   cpuTeam     = teams_[0];

    cpuTeam->startTask(cpuTask, nCpuThreads, "CpuTask", cpuTaskName);

    Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
    for (amrex::MFIter  itor(grid->unk()); itor.isValid(); ++itor) {
        W  work(itor);
        cpuTeam->enqueue(work);
    }
    grid = nullptr;
    cpuTeam->closeTask();
    cpuTeam->wait();
}

