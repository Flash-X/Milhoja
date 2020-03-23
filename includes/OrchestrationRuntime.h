/**
 * \file    OrchestrationRuntime.h
 *
 * \brief 
 *
 */

#ifndef ORCHESTRATION_RUNTIME_H__
#define ORCHESTRATION_RUNTIME_H__

#include <string>

#include "ThreadTeam.h"
#include "runtimeTask.h"

template<typename W>
class OrchestrationRuntime {
public:
    ~OrchestrationRuntime(void);

    static OrchestrationRuntime* instance(void);
    static void setLogFilename(const std::string& filename);
    static void setNumberThreadTeams(const unsigned int nTeams);
    static void setMaxThreadsPerTeam(const unsigned int maxThreads);

    void executeTasks(const std::string& bundleName,
                      TASK_FCN<W> cpuTask,
                      const unsigned int nCpuThreads,
                      const std::string& cpuTaskName,
                      TASK_FCN<W> gpuTask, 
                      const unsigned int nGpuThreads,
                      const std::string& gpuTaskName, 
                      TASK_FCN<W> postGpuTask,
                      const unsigned int nPostGpuThreads,
                      const std::string& postGpuTaskName);

private:
    OrchestrationRuntime(void);

    OrchestrationRuntime(OrchestrationRuntime&) = delete;
    OrchestrationRuntime(const OrchestrationRuntime&) = delete;
    OrchestrationRuntime(OrchestrationRuntime&&) = delete;
    OrchestrationRuntime(const OrchestrationRuntime&&) = delete;

    OrchestrationRuntime& operator=(OrchestrationRuntime&) = delete;
    OrchestrationRuntime& operator=(const OrchestrationRuntime&) = delete;
    OrchestrationRuntime& operator=(OrchestrationRuntime&&) = delete;
    OrchestrationRuntime& operator=(const OrchestrationRuntime&&) = delete;

    void executeCpuTask(const std::string& bundleName,
                        TASK_FCN<W> cpuTask,
                        const unsigned int nCpuThreads,
                        const std::string& cpuTaskName);

    void executeTasks_Full(const std::string& bundleName,
                           TASK_FCN<W> cpuTask,
                           const unsigned int nCpuThreads,
                           const std::string& cpuTaskName,
                           TASK_FCN<W> gpuTask, 
                           const unsigned int nGpuThreads,
                           const std::string& gpuTaskName, 
                           TASK_FCN<W> postGpuTask,
                           const unsigned int nPostGpuThreads,
                           const std::string& postGpuTaskName);

    static std::string             logFilename_;
    static unsigned int            nTeams_; 
    static unsigned int            maxThreadsPerTeam_;
    static OrchestrationRuntime*   instance_;

    ThreadTeam<W>**                teams_;

#ifdef DEBUG_RUNTIME
    std::ofstream     logFile_; 
#endif
};

#include "../src/OrchestrationRuntime.cpp"

#endif

