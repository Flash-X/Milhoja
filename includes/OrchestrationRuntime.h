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
#include "DataPacket.h"
#include "ThreadTeam.h"
#include "ActionBundle.h"
#include "RuntimeAction.h"

namespace orchestration {

class Runtime {
public:
    ~Runtime(void);

    static Runtime& instance(void);
    static void setLogFilename(const std::string& filename);
    static void setNumberThreadTeams(const unsigned int nTileTeams,
                                     const unsigned int nPacketTeams);
    static void setMaxThreadsPerTeam(const unsigned int maxThreads);

    void executeTasks(const ActionBundle& bundle);

private:
    Runtime(void);

    Runtime(Runtime&) = delete;
    Runtime(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;

    Runtime& operator=(Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    void executeCpuTasks(const std::string& bundleName,
                         const RuntimeAction& cpuAction);

    void executeGpuTasks(const std::string& bundleName,
                         const RuntimeAction& gpuAction);

    void executeConcurrentCpuGpuTasks(const std::string& bundleName,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction);

    void executeTasks_Full(const std::string& bundleName,
                           const RuntimeAction& cpuAction,
                           const RuntimeAction& gpuAction,
                           const RuntimeAction& postGpuAction);

    void executeTasks_FullPacket(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction);

    static unsigned int            nTileTeams_; 
    static unsigned int            nPacketTeams_; 
    static unsigned int            maxThreadsPerTeam_;
    static bool                    instantiated_;

    ThreadTeam<Tile>**             tileTeams_;
    ThreadTeam<DataPacket>**       packetTeams_;
};

}

#endif

