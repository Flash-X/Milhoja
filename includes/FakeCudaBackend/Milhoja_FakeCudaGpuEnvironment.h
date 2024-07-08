/**
 * \file    Milhoja_FakeCudaGpuEnvironment.h
 *
 * \brief 
 */

#ifndef MILHOJA_FAKECUDA_GPU_ENVIRONMENT_H__
#define MILHOJA_FAKECUDA_GPU_ENVIRONMENT_H__

#include <string>

#include "Milhoja.h"

#ifndef MILHOJA_HOSTMEM_RUNTIME_BACKEND
#error "This file need not be compiled if the FAKECUDA backend isn't used"
#endif

namespace milhoja {

class FakeCudaGpuEnvironment {
public:
    ~FakeCudaGpuEnvironment(void);

    FakeCudaGpuEnvironment(FakeCudaGpuEnvironment&)                  = delete;
    FakeCudaGpuEnvironment(const FakeCudaGpuEnvironment&)            = delete;
    FakeCudaGpuEnvironment(FakeCudaGpuEnvironment&&)                 = delete;
    FakeCudaGpuEnvironment& operator=(FakeCudaGpuEnvironment&)       = delete;
    FakeCudaGpuEnvironment& operator=(const FakeCudaGpuEnvironment&) = delete;
    FakeCudaGpuEnvironment& operator=(FakeCudaGpuEnvironment&&)      = delete;

    static void                 initialize(void);
    static FakeCudaGpuEnvironment&  instance(void);
    void                        finalize(void);

    std::size_t   nFakeDevices(void) const         { return nDevices_; }
    std::size_t   bytesInDeviceMemory(void) const { return gpuTotalGlobalMemBytes_; }

    std::string   information(void) const;

private:
    FakeCudaGpuEnvironment(void);

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

