/**
 * \file    Milhoja_FakeCudaMemoryManager.h
 *
 * \brief  Write this
 *
 * Write this.
 *
 */

#ifndef MILHOJA_FAKECUDA_MEMORY_MANAGER_H__
#define MILHOJA_FAKECUDA_MEMORY_MANAGER_H__

#include <cuchar>
#include <pthread.h>

#include "Milhoja.h"

#ifndef MILHOJA_HOSTMEM_RUNTIME_BACKEND
#error "This file need not be compiled if the HOSTMEM backend isn't used"
#endif

namespace milhoja {

class FakeCudaMemoryManager {
public:
    ~FakeCudaMemoryManager(void);

    FakeCudaMemoryManager(FakeCudaMemoryManager&)                  = delete;
    FakeCudaMemoryManager(const FakeCudaMemoryManager&)            = delete;
    FakeCudaMemoryManager(FakeCudaMemoryManager&&)                 = delete;
    FakeCudaMemoryManager& operator=(FakeCudaMemoryManager&)       = delete;
    FakeCudaMemoryManager& operator=(const FakeCudaMemoryManager&) = delete;
    FakeCudaMemoryManager& operator=(FakeCudaMemoryManager&&)      = delete;

    static void                 initialize(const std::size_t nBytesInMemoryPools);
    static FakeCudaMemoryManager&   instance(void);
    void                        finalize(void);

    void   requestMemory(const std::size_t pinnedBytes,
                         void** pinnedPtr,
                         const std::size_t gpuBytes,
                         void** gpuPtr);
    void   releaseMemory(void** pinnedPtr, void** gpuPtr);

    // FIXME: This is temprorary since this manager is so rudimentary
    void   reset(void);

private:
    FakeCudaMemoryManager(void);

    static std::size_t  nBytes_;
    static bool         initialized_;
    static bool         finalized_;

    pthread_mutex_t   mutex_;
    pthread_cond_t    memoryReleased_;

    char*         pinnedBuffer_;
    char*         gpuBuffer_;
    std::size_t   pinnedOffset_;
    std::size_t   gpuOffset_;
};

}

#endif

