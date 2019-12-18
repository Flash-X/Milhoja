#include "OrchestrationRuntime.h"

#include <stdexcept>

OrchestrationRuntime::OrchestrationRuntime(const unsigned int nMaxCpuThreads,
                                           const unsigned int nMaxGpuThreads,
                                           const unsigned int nMaxPostGpuThreads)
    : cpuTeam_(nMaxCpuThreads, "Cpu"),
      gpuTeam_(nMaxGpuThreads, "Gpu"), 
      postGpuTeam_(nMaxPostGpuThreads, "Post-Gpu") { }

OrchestrationRuntime::~OrchestrationRuntime(void) { }

void OrchestrationRuntime::executeTask(const std::vector<int>& work,
                                       TASK_FCN* cpuTask, 
                                       const unsigned int nCpuThreads,
                                       TASK_FCN* gpuTask,
                                       const unsigned int nGpuThreads,
                                       TASK_FCN* postGpuTask,
                                       const unsigned int nPostGpuThreads) {
    // TODO: The pipeline construction would be done dynamically.
    // Realistically, we would have multiple different implementations and 
    // this routine would select the setup based on the parameter values.

    unsigned int nTotalThreads = nCpuThreads + nGpuThreads + nPostGpuThreads;
    if (nTotalThreads > postGpuTeam_.nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTask] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }

    // Construct thread and work pipelines
    cpuTeam_.attachThreadReceiver(&postGpuTeam_);
    gpuTeam_.attachThreadReceiver(&postGpuTeam_);
    gpuTeam_.attachWorkReceiver(&postGpuTeam_);

    cpuTeam_.startTask(cpuTask, nCpuThreads);
    gpuTeam_.startTask(gpuTask, nGpuThreads);
    postGpuTeam_.startTask(postGpuTask, nPostGpuThreads);

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    for (auto w: work) {
        // Queue the GPU work first so that it can potentially get a head start
        gpuTeam_.enqueue(w);
        cpuTeam_.enqueue(w);
    }
    gpuTeam_.closeTask();
    cpuTeam_.closeTask();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    // CPU and GPU pools are not dependent on any other pools
    //   => call these first
    cpuTeam_.wait();
    cpuTeam_.detachThreadReceiver();

    gpuTeam_.wait();
    
    // The GPU pool has no more threads or work to push to its dependents
    gpuTeam_.detachThreadReceiver();
    gpuTeam_.detachWorkReceiver();

    // Post-GPU follows GPU in the thread pool work/thread pipeline
    //   => the gpuTeam wait() method *must* terminate before the postGpuTeam
    //      wait() method is called to ensure that all GPU work are queued
    //      in postGpuTeam before the post-GPU thread pool can begin
    //      determining if it should terminate.
    postGpuTeam_.wait();
}

