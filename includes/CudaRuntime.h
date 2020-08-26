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

    static CudaRuntime&  instance(void);
    static void          setLogFilename(const std::string& filename);
    static void          setNumberThreadTeams(const unsigned int nTeams);
    static void          setMaxThreadsPerTeam(const unsigned int maxThreads);

    unsigned int numberFreeStreams(void) const;

//    void executeTasks(const ActionBundle& bundle);

    void printGpuInformation(void) const;

private:
    CudaRuntime(void);

    CudaRuntime(CudaRuntime&) = delete;
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime(CudaRuntime&&) = delete;

    CudaRuntime& operator=(CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;
    CudaRuntime& operator=(CudaRuntime&&) = delete;

//    void executeGpuTasks(const std::string& bundleName,
//                         const RuntimeAction& gpuAction);

    static unsigned int    nTeams_; 
    static unsigned int    maxThreadsPerTeam_;
    static bool            instantiated_;

    ThreadTeam**  teams_;

    std::string   gpuDeviceName_;
    int           gpuCompMajor_;
    int           gpuCompMinor_;
    int           gpuMaxGridSize_[3];
    int           gpuMaxThreadDim_[3];
    int           gpuMaxThreadsPerBlock_;
    int           gpuWarpSize_;
    double        gpuClockRateHz_;
    double        gpuMemClockRateHz_;
    int           gpuMemBusWidthBytes_;
    size_t        gpuTotalGlobalMemBytes_;
    int           gpuL2CacheSizeBytes_;
    bool          gpuSupportsL1Caching_;
    int           gpuNumMultiprocessors_;
    int           gpuMaxConcurrentKernels_;
};

}

#endif

