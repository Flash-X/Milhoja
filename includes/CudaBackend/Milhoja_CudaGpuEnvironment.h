/**
 * \file    Milhoja_CudaGpuEnvironment.h
 *
 * \brief 
 */

#ifndef MILHOJA_CUDA_GPU_ENVIRONMENT_H__
#define MILHOJA_CUDA_GPU_ENVIRONMENT_H__

#include <string>

#include "Milhoja.h"

#ifndef MILHOJA_CUDA_RUNTIME_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

namespace milhoja {

class CudaGpuEnvironment {
public:
    ~CudaGpuEnvironment(void);

    CudaGpuEnvironment(CudaGpuEnvironment&)                  = delete;
    CudaGpuEnvironment(const CudaGpuEnvironment&)            = delete;
    CudaGpuEnvironment(CudaGpuEnvironment&&)                 = delete;
    CudaGpuEnvironment& operator=(CudaGpuEnvironment&)       = delete;
    CudaGpuEnvironment& operator=(const CudaGpuEnvironment&) = delete;
    CudaGpuEnvironment& operator=(CudaGpuEnvironment&&)      = delete;

    static void                 initialize(void);
    static CudaGpuEnvironment&  instance(void);
    void                        finalize(void);

    std::size_t   nGpuDevices(void) const         { return nDevices_; }
    std::size_t   bytesInDeviceMemory(void) const { return gpuTotalGlobalMemBytes_; }

    std::string   information(void) const;

private:
    CudaGpuEnvironment(void);

    static bool   initialized_;
    static bool   finalized_;

    int           nDevices_;
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
    std::size_t   gpuTotalGlobalMemBytes_;
    int           gpuL2CacheSizeBytes_;
    bool          gpuSupportsL1Caching_;
    int           gpuNumMultiprocessors_;
    int           gpuMaxConcurrentKernels_;
};

}

#endif

