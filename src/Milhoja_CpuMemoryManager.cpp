#include "Milhoja_CpuMemoryManager.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include "Milhoja_Logger.h"

namespace milhoja {

std::size_t   CpuMemoryManager::nBytes_ = 0;
bool          CpuMemoryManager::initialized_ = false;
bool          CpuMemoryManager::finalized_ = false;

/**
 *
 * \return 
 */
void CpuMemoryManager::initialize(const std::size_t nBytesInMemoryPools) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[CpuMemoryManager::initialize] "
                               "Memory manager already initialized");
    } else if (nBytesInMemoryPools == 0) {
        throw std::invalid_argument("[CpuMemoryManager::initialize] "
                                    "Buffers must be non-empty");
    }
    // TODO: Check that buffers are sized for byte alignment?

    Logger::instance().log("[CpuMemoryManager] Initializing...");

    nBytes_ = nBytesInMemoryPools;
    initialized_ = true;

    instance();

    Logger::instance().log("[CpuMemoryManager] Created and ready for use");
    Logger::instance().log("[CpuMemoryManager] WARNING: Memory pool *NOT* implemented yet");
}

/**
 *
 */
void    CpuMemoryManager::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[CpuMemoryManager::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[CpuMemoryManager::finalize] Already finalized");
    }

    Logger::instance().log("[CpuMemoryManager] Finalizing ...");

    Logger::instance().log(  "[CpuMemoryManager] Deallocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of CPU memory");

    finalized_ = true;
    Logger::instance().log("[CpuMemoryManager] Finalized");
}

/**
 *
 * \return 
 */
CpuMemoryManager&   CpuMemoryManager::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[CpuMemoryManager::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[CpuMemoryManager::instance] No access after finalization");
    }

    static CpuMemoryManager   manager;
    return manager;
}

/**
 * 
 *
 * \return 
 */
CpuMemoryManager::CpuMemoryManager(void) {
    Logger::instance().log(  "[CpuMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of CPU memory");
}

/**
 * 
 *
 * \return 
 */
CpuMemoryManager::~CpuMemoryManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[CpuMemoryManager::~CpuMemoryManager] ERROR - Not finalized"
                  << std::endl;
    }
}

/**
 * Refer to the RuntimeBackend documentation for requestCpuMemory for more
 * information.
 *
 * @todo We should implement a generic thread-safe class that manages a CPU
 * memory pool.  Hopefully, all backends should be able to use that single
 * class.  Update this and the paired release function once done.
 */
void      CpuMemoryManager::requestMemory(const std::size_t nBytes,
                                          void** ptr) {
    if (!ptr) {
        std::string  errMsg = "[CpuMemoryManager::requestCpuMemory] ";
        errMsg += "Null handle given\n";
        throw std::invalid_argument(errMsg);
    } else if (*ptr) {
        std::string  errMsg = "[CpuMemoryManager::requestCpuMemory] ";
        errMsg += "Internal pointer already set\n";
        throw std::invalid_argument(errMsg);
    } else if (nBytes == 0) {
        std::string  errMsg = "[CpuMemoryManager::requestCpuMemory] ";
        errMsg += "Requests of zero indicate logical error\n";
        throw std::invalid_argument(errMsg);
    }

    *ptr = std::malloc(nBytes);
    std::memset(*ptr, 0, nBytes);
}

/**
 * Refer to the RuntimeBackend documentation for releaseCpuMemory more
 * information.
 */
void      CpuMemoryManager::releaseMemory(void** ptr) {
    if (!ptr) {
        std::string  errMsg = "[CpuMemoryManager::releaseCpuMemory] ";
        errMsg += "Null handle given\n";
        throw std::invalid_argument(errMsg);
    } else if (!*ptr) {
        std::string  errMsg = "[CpuMemoryManager::releaseCpuMemory] ";
        errMsg += "Internal pointer is null\n";
        throw std::invalid_argument(errMsg);
    }

    std::free(*ptr);
    *ptr = nullptr;
}

}

