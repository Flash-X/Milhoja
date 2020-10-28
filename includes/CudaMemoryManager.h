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

    static void                 setBufferSize(const std::size_t bytes);
    static CudaMemoryManager&   instance(void);

    void   requestMemory(const std::size_t bytes,
                         void** hostPtr, void** gpuPtr);
    void   releaseMemory(void** hostPtr, void** gpuPtr);

    // FIXME: This is temprorary since this manager is so rudimentary
    void   reset(void);

private:
    CudaMemoryManager(void);

    static std::size_t  nBytes_;
    static bool         wasInstantiated_;

    pthread_mutex_t   mutex_;
    pthread_cond_t    memoryReleased_;

    char*         pinnedBuffer_;
    char*         gpuBuffer_;
    std::size_t   offset_;
};

}

#endif

