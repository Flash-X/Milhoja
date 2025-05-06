#include "Milhoja_OmpTargetGpuEnvironment.h"

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>
#include <omp.h>

#include "Milhoja_Logger.h"

namespace milhoja {

bool    OmpTargetGpuEnvironment::initialized_ = false;
bool    OmpTargetGpuEnvironment::finalized_   = false;

/**
 * 
 *
 */
void   OmpTargetGpuEnvironment::initialize(void) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[OmpTargetGpuEnvironment::initialize] Already initialized");
    }

    Logger::instance().log("[OmpTargetGpuEnvironment] Initializing...");

    initialized_ = true;

    // Create/initialize environment
    const OmpTargetGpuEnvironment&   gpuEnv = instance();
    std::string   msg =   "[OmpTargetGpuEnvironment] " 
                        + std::to_string(gpuEnv.nGpuDevices()) 
                        + " GPU device(s) per process found\n"
                        + gpuEnv.information();
    Logger::instance().log(msg);
    Logger::instance().log("[OmpTargetGpuEnvironment] Created and ready for use");
}

/**
 *
 */
void    OmpTargetGpuEnvironment::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[OmpTargetGpuEnvironment::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[OmpTargetGpuEnvironment::finalize] Already finalized");
    }

    Logger::instance().log("[OmpTargetGpuEnvironment] Finalizing ...");

    finalized_ = true;

    Logger::instance().log("[OmpTargetGpuEnvironment] Finalized");
}

/**
 * 
 *
 * \return 
 */
OmpTargetGpuEnvironment& OmpTargetGpuEnvironment::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[OmpTargetGpuEnvironment::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[OmpTargetGpuEnvironment::instance] No access after finalization");
    }

    static OmpTargetGpuEnvironment     singleton;
    return singleton;
}

/**
 * 
 *
 */
OmpTargetGpuEnvironment::OmpTargetGpuEnvironment(void)
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
    nDevices_ = omp_get_num_devices();
    std::string   msg =   "[OmpTargetGpuEnvironment] " 
                        + std::to_string(nDevices_) 
                        + " was returned from omp_get_num_devices().\n";
    Logger::instance().log(msg);
    if (nDevices_ != 1 && nDevices_ != 0) {
        throw std::runtime_error("[OmpTargetGpuEnvironment::OmpTargetGpuEnvironment] "
                                 "We insist on 1 or 0 target devices per MPI task");
    }

    // cudaDeviceProp  prop;
    // cudaGetDeviceProperties(&prop, 0);

    // if (prop.concurrentKernels != 1) {
    //     throw std::runtime_error("[OmpTargetGpuEnvironment::OmpTargetGpuEnvironment] "
    //                              "GPU kernel concurrency is required");
    // }

    gpuDeviceName_          = std::string("OpenMP target device");
    // gpuCompMajor_           = prop.major;
    // gpuCompMinor_           = prop.minor;
    // gpuMaxThreadsPerBlock_  = prop.maxThreadsPerBlock;
    // gpuWarpSize_            = prop.warpSize;
    // gpuClockRateHz_         = prop.clockRate * 1000;
    // gpuMemClockRateHz_      = prop.memoryClockRate * 1000;
    // gpuMemBusWidthBytes_    = round(prop.memoryBusWidth * 0.125);
    gpuTotalGlobalMemBytes_ = 8.0*std::pow(1024.0, 3.0);
    // gpuTotalGlobalMemBytes_ = prop.totalGlobalMem;
    // gpuL2CacheSizeBytes_    = prop.l2CacheSize;
    // gpuSupportsL1Caching_   = (prop.localL1CacheSupported == 1);
    // gpuNumMultiprocessors_  = prop.multiProcessorCount;
    // for (unsigned int i=0; i<3; ++i) {
    //     gpuMaxGridSize_[i]  = prop.maxGridSize[i];
    //     gpuMaxThreadDim_[i] = prop.maxThreadsDim[i];
    // }

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    gpuMaxConcurrentKernels_ = 128;
    Logger::instance().log("[OmpTargetGpuEnvironment] Faking gpuMaxConcurrentKernels_ = 128 ...");
}

std::string  OmpTargetGpuEnvironment::information(void) const {
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
         << (gpuTotalGlobalMemBytes_ / std::pow(1024.0, 3.0)) << " GB (fake!)\n";
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

/**
 *
 */
OmpTargetGpuEnvironment::~OmpTargetGpuEnvironment(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[OmpTargetGpuEnvironment::~OmpTargetGpuEnvironment] ERROR - Not finalized"
                  << std::endl;
    }
}

}
