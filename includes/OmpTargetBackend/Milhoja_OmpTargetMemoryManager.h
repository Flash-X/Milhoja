/**
 * \file    Milhoja_OmpTargetMemoryManager.h
 *
 * \brief  Write this
 *
 * Write this.
 *
 */

#ifndef MILHOJA_OMPTARGET_MEMORY_MANAGER_H__
#define MILHOJA_OMPTARGET_MEMORY_MANAGER_H__

#include <pthread.h>

#include "Milhoja.h"

#ifndef MILHOJA_OMPTARGET_RUNTIME_BACKEND
#error "This file need not be compiled if the OmpTarget backend isn't used"
#endif

#include <omp.h>

#ifdef ORCHA_USE_OMP_REQ
#pragma omp requires unified_address
#endif

namespace milhoja {

class OmpTargetMemoryManager {
public:
    ~OmpTargetMemoryManager(void);

    OmpTargetMemoryManager(OmpTargetMemoryManager&)                  = delete;
    OmpTargetMemoryManager(const OmpTargetMemoryManager&)            = delete;
    OmpTargetMemoryManager(OmpTargetMemoryManager&&)                 = delete;
    OmpTargetMemoryManager& operator=(OmpTargetMemoryManager&)       = delete;
    OmpTargetMemoryManager& operator=(const OmpTargetMemoryManager&) = delete;
    OmpTargetMemoryManager& operator=(OmpTargetMemoryManager&&)      = delete;

    static void                 initialize(const std::size_t nBytesInMemoryPools,
					   const int target_device_num);
    static OmpTargetMemoryManager&   instance(void);
    void                        finalize(void);

    void   requestMemory(const std::size_t pinnedBytes,
                         void** pinnedPtr,
                         const std::size_t gpuBytes,
                         void** gpuPtr);
    void   releaseMemory(void** pinnedPtr, void** gpuPtr);

    // FIXME: This is temprorary since this manager is so rudimentary
    void   reset(void);

private:
    OmpTargetMemoryManager(void);

    static std::size_t  nBytes_;
    static bool         initialized_;
    static bool         finalized_;
    static int          device_num_;

    static omp_allocator_handle_t pinned_allocator_;

    pthread_mutex_t   mutex_;
    pthread_cond_t    memoryReleased_;

    char*         pinnedBuffer_;
    char*         gpuBuffer_;
    std::size_t   pinnedOffset_;
    std::size_t   gpuOffset_;

};

}

#endif

