/**
 * \file    OrchestrationRuntime.h
 *
 * \brief 
 *
 */

#ifndef ORCHESTRATION_RUNTIME_H__
#define ORCHESTRATION_RUNTIME_H__

#include <string>

#include "Tile.h"
#include "ThreadTeam.h"
#include "ActionBundle.h"
#include "RuntimeAction.h"

class OrchestrationRuntime {
public:
    ~OrchestrationRuntime(void);

    static OrchestrationRuntime* instance(void);
    static void setLogFilename(const std::string& filename);
    static void setNumberThreadTeams(const unsigned int nTeams);
    static void setMaxThreadsPerTeam(const unsigned int maxThreads);

    void executeTasks(const ActionBundle& bundle);

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

    void executeCpuTasks(const std::string& bundleName,
                         const RuntimeAction& cpuAction);

    void executeConcurrentCpuGpuTasks(const std::string& bundleName,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction);

    void executeTasks_Full(const std::string& bundleName,
                           const RuntimeAction& cpuAction,
                           const RuntimeAction& gpuAction,
                           const RuntimeAction& postGpuAction);

    static std::string             logFilename_;
    static unsigned int            nTeams_; 
    static unsigned int            maxThreadsPerTeam_;
    static OrchestrationRuntime*   instance_;

    ThreadTeam<Tile>**             teams_;

#ifdef DEBUG_RUNTIME
    std::ofstream     logFile_; 
#endif
};

#endif

