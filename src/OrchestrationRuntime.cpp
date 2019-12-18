#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

OrchestrationRuntime::OrchestrationRuntime(const unsigned int nTeams,
                                           const unsigned int nMaxThreads)
    : nTeams_(nTeams),
      teams_(nullptr)
{
    if (nTeams_ == 0) {
        throw std::invalid_argument("[OrchestrationRuntime::OrchestrationRuntime] "
                                    "Need at least one ThreadTeam");
    }

    std::cout << "[OrchestrationRuntime] Initializing\n";
    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(nMaxThreads, i);
    }
    std::cout << "[OrchestrationRuntime] Initialized\n";
}

OrchestrationRuntime::~OrchestrationRuntime(void) {
    std::cout << "[OrchestrationRuntime] Finalizing\n";
    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }

    delete [] teams_;
    teams_ = nullptr;
    std::cout << "[OrchestrationRuntime] Destroyed\n";
}

void OrchestrationRuntime::executeTask(const std::vector<int>& work,
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

    ThreadTeam*   cpuTeam     = teams_[0];
    ThreadTeam*   gpuTeam     = teams_[1];
    ThreadTeam*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads = nCpuThreads + nGpuThreads + nPostGpuThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTask] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }

    std::cout << "[OrchestrationRuntime] Start execution of " 
              << bundleName << std::endl;

    // Construct thread and work pipelines
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
    for (auto w: work) {
        // Queue the GPU work first so that it can potentially get a head start
        gpuTeam->enqueue(w);
        cpuTeam->enqueue(w);
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

    std::cout << "[OrchestrationRuntime] Finished execution of " 
              << bundleName << std::endl;
}

