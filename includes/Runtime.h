/**
 * \file    Runtime.h
 *
 * \brief 
 *
 */

#ifndef RUNTIME_H__
#define RUNTIME_H__

#include <string>

#include "ThreadTeam.h"
#include "ActionBundle.h"
#include "RuntimeAction.h"

#if defined(USE_CUDA_BACKEND)
#include "MoverUnpacker.h"
#endif

namespace orchestration {

class Runtime {
public:
    ~Runtime(void);

    Runtime(Runtime&)                  = delete;
    Runtime(const Runtime&)            = delete;
    Runtime(Runtime&&)                 = delete;
    Runtime& operator=(Runtime&)       = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime& operator=(Runtime&&)      = delete;

    static void          instantiate(const unsigned int nTeams,
                                     const unsigned int nThreadsPerTeam,
                                     const unsigned int nStreams,
                                     const std::size_t nBytesInMemoryPools);
    static Runtime&      instance(void);

    void executeTasks(const ActionBundle& bundle);

    void executeCpuTasks(const std::string& actionName,
                         const RuntimeAction& cpuAction);
#if defined(USE_CUDA_BACKEND)
    void executeGpuTasks(const std::string& actionName,
                         const RuntimeAction& gpuAction);
    void executeCpuGpuTasks(const std::string& bundleName,
                            const RuntimeAction& cpuAction,
                            const RuntimeAction& gpuAction);
    void executeCpuGpuSplitTasks(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const unsigned int nTilesPerCpuTurn);
    void executeCpuGpuWowzaTasks(const std::string& bundleName,
                                 const RuntimeAction& actionA_cpu,
                                 const RuntimeAction& actionA_gpu,
                                 const RuntimeAction& actionB_gpu,
                                 const unsigned int nTilesPerCpuTurn);
    void executeTasks_FullPacket(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction);
#endif

private:
    Runtime(void);

    static unsigned int    nTeams_; 
    static unsigned int    maxThreadsPerTeam_;
    static bool            instantiated_;

    ThreadTeam**     teams_;

#if defined(USE_CUDA_BACKEND)
    MoverUnpacker    gpuToHost1_;
    MoverUnpacker    gpuToHost2_;
#endif
};

}

#endif
