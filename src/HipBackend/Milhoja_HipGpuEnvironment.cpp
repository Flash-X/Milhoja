#include "Milhoja_HipGpuEnvironment.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "Milhoja_Logger.h"

namespace milhoja {

bool HipGpuEnvironment::initialized_ = false;
bool HipGpuEnvironment::finalized_ = false;

void HipGpuEnvironment::initialize(void) {
    if (initialized_) {
        throw std::logic_error("[HipGpuEnvironment::initialize] Already initialized");
    }
    initialized_ = true;
    const HipGpuEnvironment& environment = instance();
    Logger::instance().log(
        "[HipGpuEnvironment] " +
        std::to_string(environment.nGpuDevices()) +
        " GPU device(s) per process found\n" + environment.information()
    );
}

void HipGpuEnvironment::finalize(void) {
    if (!initialized_) {
        throw std::logic_error("[HipGpuEnvironment::finalize] Never initialized");
    }
    if (finalized_) {
        throw std::logic_error("[HipGpuEnvironment::finalize] Already finalized");
    }
    finalized_ = true;
}

HipGpuEnvironment& HipGpuEnvironment::instance(void) {
    if (!initialized_) {
        throw std::logic_error("[HipGpuEnvironment::instance] Not initialized");
    }
    if (finalized_) {
        throw std::logic_error("[HipGpuEnvironment::instance] Already finalized");
    }
    static HipGpuEnvironment singleton;
    return singleton;
}

HipGpuEnvironment::HipGpuEnvironment(void)
    : nDevices_{0},
      deviceName_{},
      architectureName_{},
      totalGlobalMemory_{0},
      multiprocessorCount_{0},
      warpSize_{0} {
    hipError_t error = hipGetDeviceCount(&nDevices_);
    if (error != hipSuccess) {
        throw std::runtime_error(
            "[HipGpuEnvironment] hipGetDeviceCount: " +
            std::string(hipGetErrorString(error))
        );
    }
    if (nDevices_ != 1) {
        throw std::runtime_error(
            "[HipGpuEnvironment] Exactly one visible GPU per MPI task is required"
        );
    }

    error = hipSetDevice(0);
    if (error != hipSuccess) {
        throw std::runtime_error(
            "[HipGpuEnvironment] hipSetDevice: " +
            std::string(hipGetErrorString(error))
        );
    }

    hipDeviceProp_t properties{};
    error = hipGetDeviceProperties(&properties, 0);
    if (error != hipSuccess) {
        throw std::runtime_error(
            "[HipGpuEnvironment] hipGetDeviceProperties: " +
            std::string(hipGetErrorString(error))
        );
    }
    deviceName_ = properties.name;
    architectureName_ = properties.gcnArchName;
    totalGlobalMemory_ = properties.totalGlobalMem;
    multiprocessorCount_ = properties.multiProcessorCount;
    warpSize_ = properties.warpSize;
}

std::string HipGpuEnvironment::information(void) const {
    std::ostringstream information;
    information << "  Name                    " << deviceName_ << "\n"
                << "  Architecture            " << architectureName_ << "\n"
                << "  Total Global Memory     " << totalGlobalMemory_ << " bytes\n"
                << "  Multiprocessors         " << multiprocessorCount_ << "\n"
                << "  Warp/Wavefront Size     " << warpSize_ << "\n";
    return information.str();
}

HipGpuEnvironment::~HipGpuEnvironment(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[HipGpuEnvironment::~HipGpuEnvironment] ERROR - Not finalized\n";
    }
}

}
