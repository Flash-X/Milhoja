#ifndef ORCHESTRATION_RUNTIME_H__
#define ORCHESTRATION_RUNTIME_H__

#include "ThreadTeam.h"
#include "runtimeTask.h"

class OrchestrationRuntime {
public:
    OrchestrationRuntime(const unsigned int nTeams,
                         const unsigned int nMaxThreads);
    ~OrchestrationRuntime(void);

    void executeTask(const std::vector<int>& work,
                     const std::string& bundleName,
                     TASK_FCN* cpuTask,
                     const unsigned int nCpuThreads,
                     const std::string& cpuTaskName,
                     TASK_FCN* gpuTask, 
                     const unsigned int nGpuThreads,
                     const std::string& gpuTaskName, 
                     TASK_FCN* postGpuTask,
                     const unsigned int nPostGpuThreads,
                     const std::string& postGpuTaskName);

protected:

private:
    unsigned int  nTeams_;
    ThreadTeam**  teams_;
};

#endif

