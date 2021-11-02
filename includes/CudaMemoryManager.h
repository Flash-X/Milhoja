/**
 * \file    CudaMemoryManager.h
 *
 * \brief  Write this
 *
 * Write this.
 *
 */

#ifndef CUDA_MEMORY_MANAGER_H__
#define CUDA_MEMORY_MANAGER_H__

#include <pthread.h>

namespace orchestration {

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

    // Specify byte alignment of each memory request
    static constexpr std::size_t    ALIGN_SIZE = 256;
    static constexpr std::size_t    pad(const std::size_t sz) 
        { return ((sz + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE; }

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

