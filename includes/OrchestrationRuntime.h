/**
 * \file    OrchestrationRuntime.h
 *
 * \brief 
 *
 */

#ifndef ORCHESTRATION_RUNTIME_H__
#define ORCHESTRATION_RUNTIME_H__

#include "Grid.h"
#include "ThreadTeam.h"
#include "runtimeTask.h"

class OrchestrationRuntime {
public:
    ~OrchestrationRuntime(void);

    static OrchestrationRuntime* instance(void);
    static void setNumberThreadTeams(const unsigned int nTeams);
    static void setMaxThreadsPerTeam(const unsigned int maxThreads);

    void executeTask(Grid& myGrid,
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

private:
    OrchestrationRuntime(void);
    OrchestrationRuntime(const OrchestrationRuntime&);
    OrchestrationRuntime& operator=(const OrchestrationRuntime&);

    static unsigned int            nTeams_; 
    static unsigned int            maxThreadsPerTeam_;
    static OrchestrationRuntime*   instance_;

    ThreadTeam**                   teams_;
};

#endif

