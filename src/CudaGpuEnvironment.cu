#ifndef USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

#include "CudaGpuEnvironment.h"

#include <stdexcept>
#include <sstream>

#include "OrchestrationLogger.h"

namespace orchestration {

bool    CudaGpuEnvironment::instantiated_ = false;

/**
 * 
 *
 * \return 
 */
void   CudaGpuEnvironment::instantiate(void) {
    Logger::instance().log("[CudaGpuEnvironment] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[CudaGpuEnvironment::instantiate] Already instantiated");
    }
    instantiated_ = true;

    // Create/initialize environment
    const CudaGpuEnvironment&   gpuEnv = instance();
    std::string   msg =   "[CudaGpuEnvironment] " 
                        + std::to_string(gpuEnv.nGpuDevices()) 
                        + " GPU device(s) per process found\n"
                        + gpuEnv.information();
    Logger::instance().log(msg);
    Logger::instance().log("[CudaGpuEnvironment] Created and ready for use");
}

/**
 * 
 *
 * \return 
 */
CudaGpuEnvironment& CudaGpuEnvironment::instance(void) {
    static CudaGpuEnvironment     singleton;
    return singleton;
}

/**
 * 
 *
 * \return 
 */
CudaGpuEnvironment::CudaGpuEnvironment(void)
    : nDevices_{0},
      gpuDeviceName_{""},
      gpuCompMajor_{-1},
      gpuCompMinor_{-1},
      gpuMaxGridSize_{-1, -1, -1},
      gpuMaxThreadDim_{-1, -1, -1},
      gpuMaxThreadsPerBlock_{-1},
      gpuWarpSize_{-1},
      gpuClockRateHz_{-1.0},
      gpuMemClockRateHz_{-1.0},
      gpuMemBusWidthBytes_{-1},
      gpuTotalGlobalMemBytes_{0},
      gpuL2CacheSizeBytes_{-1},
      gpuSupportsL1Caching_{false},
      gpuNumMultiprocessors_{-1},
      gpuMaxConcurrentKernels_{-1}
{
    cudaGetDeviceCount(&nDevices_);
    if (nDevices_ != 1) {
        throw std::runtime_error("[CudaGpuEnvironment::CudaGpuEnvironment] "
                                 "We insist upon 1 GPU per MPI task");
    }

    cudaDeviceProp  prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.concurrentKernels != 1) {
        throw std::runtime_error("[CudaGpuEnvironment::CudaGpuEnvironment] "
                                 "GPU kernel concurrency is required");
    }

    gpuDeviceName_          = std::string(prop.name);
    gpuCompMajor_           = prop.major;
    gpuCompMinor_           = prop.minor;
    gpuMaxThreadsPerBlock_  = prop.maxThreadsPerBlock;
    gpuWarpSize_            = prop.warpSize;
    gpuClockRateHz_         = prop.clockRate * 1000;
    gpuMemClockRateHz_      = prop.memoryClockRate * 1000;
    gpuMemBusWidthBytes_    = round(prop.memoryBusWidth * 0.125);
    gpuTotalGlobalMemBytes_ = prop.totalGlobalMem;
    gpuL2CacheSizeBytes_    = prop.l2CacheSize;
    gpuSupportsL1Caching_   = (prop.localL1CacheSupported == 1);
    gpuNumMultiprocessors_  = prop.multiProcessorCount;
    for (unsigned int i=0; i<3; ++i) {
        gpuMaxGridSize_[i]  = prop.maxGridSize[i];
        gpuMaxThreadDim_[i] = prop.maxThreadsDim[i];
    }

    if (gpuCompMajor_ != 7) {
        throw std::runtime_error("[CudaGpuEnvironment::CudaGpuEnvironment] "
                                 "We assume GPU compute capability 7.X");
    }

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    gpuMaxConcurrentKernels_ = 128;
}

std::string  CudaGpuEnvironment::information(void) const {
    std::stringstream     info;
    info << "  Name                    "
         <<  gpuDeviceName_ << "\n";
    info << "  Clock Rate              "
         << (gpuClockRateHz_ * 1.0e-9) << " GHz\n";
    info << "  Memory Clock Rate       "
         << (gpuMemClockRateHz_ * 1.0e-9) << " GHz\n";
    info << "  Memory Bus Width        "
         <<  gpuMemBusWidthBytes_ << " bytes\n";
    info << "  Total Global Memory     "
         << (gpuTotalGlobalMemBytes_ / std::pow(1024.0, 3.0)) << " GB\n";
    info << "  L2 Cache Size           "
         << (gpuL2CacheSizeBytes_ / std::pow(1024.0, 2.0)) << " MB\n";
    info << "  Supports local L1 Cache "
         <<  (gpuSupportsL1Caching_ ? 'T' : 'F') << "\n";
    info << "  Compute Capability      "
         <<  gpuCompMajor_ << "." << gpuCompMinor_ << "\n";
    info << "  Max Grid Size           "
         <<  gpuMaxGridSize_[0] << " x "
         <<  gpuMaxGridSize_[1] << " x "
         <<  gpuMaxGridSize_[2] << "\n";
    info << "  Max Thread Dims         "
         <<  gpuMaxThreadDim_[0] << " x "
         <<  gpuMaxThreadDim_[1] << " x "
         <<  gpuMaxThreadDim_[2] << "\n";
    info << "  Max Threads/Block       "
         <<  gpuMaxThreadsPerBlock_ << "\n";
    info << "  Warp Size               "
         <<  gpuWarpSize_ << "\n";
    info << "  Num Multiprocessors     "
         <<  gpuNumMultiprocessors_ << "\n";
    info << "  Max Concurrent Kernels  "
         <<  gpuMaxConcurrentKernels_;

    return info.str();
}

}

