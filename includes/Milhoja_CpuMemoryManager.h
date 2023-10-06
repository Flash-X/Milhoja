#ifndef MILHOJA_CPU_MEMORY_MANAGER_H__
#define MILHOJA_CPU_MEMORY_MANAGER_H__

#include <cstddef>

namespace milhoja {

/**
 * \class CpuMemoryManager Milhoja_CpuMemoryManager.h
 *
 * Provide a generic, concrete implementation of a class that constructs and
 * manages a memory pool in the CPU memory system.  It is intended that this
 * function directly for all Runtime backends.
 *
 * @todo Implement a proper memory pool.
 */
class CpuMemoryManager {
public:
    ~CpuMemoryManager(void);

    CpuMemoryManager(CpuMemoryManager&)                  = delete;
    CpuMemoryManager(const CpuMemoryManager&)            = delete;
    CpuMemoryManager(CpuMemoryManager&&)                 = delete;
    CpuMemoryManager& operator=(CpuMemoryManager&)       = delete;
    CpuMemoryManager& operator=(const CpuMemoryManager&) = delete;
    CpuMemoryManager& operator=(CpuMemoryManager&&)      = delete;

    static void                initialize(const std::size_t nBytesInMemoryPool);
    static CpuMemoryManager&   instance(void);
    void                       finalize(void);

    void      requestMemory(const std::size_t nBytes, void** ptr);
    void      releaseMemory(void** ptr);

private:
    CpuMemoryManager(void);

    static std::size_t  nBytes_;
    static bool         initialized_;
    static bool         finalized_;
};

}

#endif

