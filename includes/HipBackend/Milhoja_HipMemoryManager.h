#ifndef MILHOJA_HIP_MEMORY_MANAGER_H__
#define MILHOJA_HIP_MEMORY_MANAGER_H__

#include <cstddef>
#include <pthread.h>

#include "Milhoja.h"

#ifndef MILHOJA_HIP_RUNTIME_BACKEND
#error "This file need not be compiled if the HIP backend is not used"
#endif

namespace milhoja {

class HipMemoryManager {
public:
    ~HipMemoryManager(void);

    HipMemoryManager(HipMemoryManager&) = delete;
    HipMemoryManager(const HipMemoryManager&) = delete;
    HipMemoryManager(HipMemoryManager&&) = delete;
    HipMemoryManager& operator=(HipMemoryManager&) = delete;
    HipMemoryManager& operator=(const HipMemoryManager&) = delete;
    HipMemoryManager& operator=(HipMemoryManager&&) = delete;

    static void initialize(const std::size_t nBytesInMemoryPools);
    static HipMemoryManager& instance(void);
    void finalize(void);

    void requestMemory(const std::size_t pinnedBytes,
                       void** pinnedPtr,
                       const std::size_t gpuBytes,
                       void** gpuPtr);
    void releaseMemory(void** pinnedPtr, void** gpuPtr);
    void reset(void);

private:
    HipMemoryManager(void);

    static std::size_t nBytes_;
    static bool initialized_;
    static bool finalized_;

    pthread_mutex_t mutex_;
    char* pinnedBuffer_;
    char* gpuBuffer_;
    std::size_t pinnedOffset_;
    std::size_t gpuOffset_;
};

}

#endif
