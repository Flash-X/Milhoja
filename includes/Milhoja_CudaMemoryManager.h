/**
 * \file    Milhoja_CudaMemoryManager.h
 *
 * \brief  Write this
 *
 * Write this.
 *
 */

#ifndef MILHOJA_CUDA_MEMORY_MANAGER_H__
#define MILHOJA_CUDA_MEMORY_MANAGER_H__

#include <pthread.h>

namespace milhoja {

class CudaMemoryManager {
public:
    ~CudaMemoryManager(void);

    CudaMemoryManager(CudaMemoryManager&)                  = delete;
    CudaMemoryManager(const CudaMemoryManager&)            = delete;
    CudaMemoryManager(CudaMemoryManager&&)                 = delete;
    CudaMemoryManager& operator=(CudaMemoryManager&)       = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(CudaMemoryManager&&)      = delete;

    static void                 instantiate(const std::size_t nBytesInMemoryPools);
    static CudaMemoryManager&   instance(void);

    void   requestMemory(const std::size_t pinnedBytes,
                         void** pinnedPtr,
                         const std::size_t gpuBytes,
                         void** gpuPtr);
    void   releaseMemory(void** pinnedPtr, void** gpuPtr);

    // FIXME: This is temprorary since this manager is so rudimentary
    void   reset(void);

private:
    CudaMemoryManager(void);

    static std::size_t  nBytes_;
    static bool         instantiated_;

    pthread_mutex_t   mutex_;
    pthread_cond_t    memoryReleased_;

    char*         pinnedBuffer_;
    char*         gpuBuffer_;
    std::size_t   pinnedOffset_;
    std::size_t   gpuOffset_;
};

}

#endif

