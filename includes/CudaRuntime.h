/**
 * \file    CudaRuntime.h
 *
 * \brief 
 *
 */

#ifndef CUDA_RUNTIME_H__
#define CUDA_RUNTIME_H__

#include <string>

#include "ThreadTeam.h"
#include "ActionBundle.h"
#include "RuntimeAction.h"

namespace orchestration {

class CudaRuntime {
public:
    ~CudaRuntime(void);

    CudaRuntime(CudaRuntime&)                  = delete;
    CudaRuntime(const CudaRuntime&)            = delete;
    CudaRuntime(CudaRuntime&&)                 = delete;
    CudaRuntime& operator=(CudaRuntime&)       = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;
    CudaRuntime& operator=(CudaRuntime&&)      = delete;

    static void          setLogFilename(const std::string& filename);
    static void          setNumberThreadTeams(const unsigned int nTeams);
    static void          setMaxThreadsPerTeam(const unsigned int maxThreads);
    static CudaRuntime&  instance(void);

    unsigned int numberFreeStreams(void) const;

//    void executeTasks(const ActionBundle& bundle);

    void executeCpuTasks(const std::string& actionName,
                         const RuntimeAction& cpuAction);
    void executeGpuTasks(const std::string& actionName,
                         const RuntimeAction& gpuAction);
    void executeTasks_FullPacket(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction);

private:
    CudaRuntime(void);

    static unsigned int    nTeams_; 
    static unsigned int    maxThreadsPerTeam_;
    static bool            instantiated_;

    ThreadTeam**  teams_;
};

}

#endif

