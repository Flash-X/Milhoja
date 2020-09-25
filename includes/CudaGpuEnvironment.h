/**
 * \file    CudaGpuEnvironment.h
 *
 * \brief 
 */

#ifndef CUDA_GPU_ENVIRONMENT_H__
#define CUDA_GPU_ENVIRONMENT_H__

#include <string>

namespace orchestration {

class CudaGpuEnvironment {
public:
    ~CudaGpuEnvironment(void);

    CudaGpuEnvironment(CudaGpuEnvironment&)                  = delete;
    CudaGpuEnvironment(const CudaGpuEnvironment&)            = delete;
    CudaGpuEnvironment(CudaGpuEnvironment&&)                 = delete;
    CudaGpuEnvironment& operator=(CudaGpuEnvironment&)       = delete;
    CudaGpuEnvironment& operator=(const CudaGpuEnvironment&) = delete;
    CudaGpuEnvironment& operator=(CudaGpuEnvironment&&)      = delete;

    static CudaGpuEnvironment&  instance(void);

    std::size_t   nGpuDevices(void) const         { return nDevices_; }
    std::size_t   bytesInDeviceMemory(void) const { return gpuTotalGlobalMemBytes_; }

    std::string   information(void) const;

private:
    CudaGpuEnvironment(void);

    static bool            instantiated_;

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

