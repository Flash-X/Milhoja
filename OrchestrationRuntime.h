#ifndef ORCHESTRATION_RUNTIME_H__
#define ORCHESTRATION_RUNTIME_H__

#include "ThreadTeam.h"
#include "runtimeTask.h"

class OrchestrationRuntime {
public:
    OrchestrationRuntime(const unsigned int nMaxCpuThreads,
                         const unsigned int nMaxGpuThreads,
                         const unsigned int nMaxPostGpuThreads);
    ~OrchestrationRuntime(void);

    void executeTask(const std::vector<int>& work,
                     TASK_FCN* cpuTask, const unsigned int nCpuThreads,
                     TASK_FCN* gpuTask, const unsigned int nGpuThreads,
                     TASK_FCN* postGpuTask, const unsigned int nPostGpuThreads);

protected:

private:
    ThreadTeam   cpuTeam_;
    ThreadTeam   gpuTeam_;
    ThreadTeam   postGpuTeam_;
};

#endif

