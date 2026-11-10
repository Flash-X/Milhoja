#ifndef MILHOJA_HIP_GPU_ENVIRONMENT_H__
#define MILHOJA_HIP_GPU_ENVIRONMENT_H__

#include <cstddef>
#include <string>

#include "Milhoja.h"

#ifndef MILHOJA_HIP_RUNTIME_BACKEND
#error "This file need not be compiled if the HIP backend is not used"
#endif

namespace milhoja {

class HipGpuEnvironment {
public:
    ~HipGpuEnvironment(void);

    HipGpuEnvironment(HipGpuEnvironment&) = delete;
    HipGpuEnvironment(const HipGpuEnvironment&) = delete;
    HipGpuEnvironment(HipGpuEnvironment&&) = delete;
    HipGpuEnvironment& operator=(HipGpuEnvironment&) = delete;
    HipGpuEnvironment& operator=(const HipGpuEnvironment&) = delete;
    HipGpuEnvironment& operator=(HipGpuEnvironment&&) = delete;

    static void initialize(void);
    static HipGpuEnvironment& instance(void);
    void finalize(void);

    std::size_t nGpuDevices(void) const { return nDevices_; }
    std::size_t bytesInDeviceMemory(void) const { return totalGlobalMemory_; }
    std::string information(void) const;

private:
    HipGpuEnvironment(void);

    static bool initialized_;
    static bool finalized_;

    int nDevices_;
    std::string deviceName_;
    std::string architectureName_;
    std::size_t totalGlobalMemory_;
    int multiprocessorCount_;
    int warpSize_;
};

}

#endif
