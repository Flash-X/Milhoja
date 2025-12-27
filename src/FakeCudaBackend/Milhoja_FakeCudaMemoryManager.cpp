#include "Milhoja_FakeCudaMemoryManager.h"

#include <stdexcept>
#include <cmath>
#include <iostream>

#include "Milhoja_Logger.h"
//#include "Milhoja_CudaGpuEnvironment.h"

namespace milhoja {

std::size_t   FakeCudaMemoryManager::nBytes_ = 0;
bool          FakeCudaMemoryManager::initialized_ = false;
bool          FakeCudaMemoryManager::finalized_ = false;

/**
 *
 */
void FakeCudaMemoryManager::initialize(const std::size_t nBytesInMemoryPools) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[FakeCudaMemoryManager::initialize] "
                               "Memory manager already initialized");
    } else if (nBytesInMemoryPools == 0) {
        throw std::invalid_argument("[FakeCudaMemoryManager::initialize] "
                                    "Buffers must be non-empty");
    }
    // TODO: Check that buffers are sized for byte alignment?

    Logger::instance().log("[FakeCudaMemoryManager] Initializing...");

    nBytes_ = nBytesInMemoryPools;
    initialized_ = true;

    instance();

    Logger::instance().log("[FakeCudaMemoryManager] Created and ready for use");
}

/**
 *
 */
void    FakeCudaMemoryManager::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[FakeCudaMemoryManager::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[FakeCudaMemoryManager::finalize] Already finalized");
    }

    Logger::instance().log("[FakeCudaMemoryManager] Finalizing ...");

    pthread_mutex_lock(&mutex_);

    if (pinnedBuffer_ != nullptr) {
      //fakeCudaError_t   cErr = fakeCudaFreeHost(pinnedBuffer_);
        free (pinnedBuffer_);
        // if (cErr != fakeCudaSuccess) {
        //     std::string  msg = "[FakeCudaMemoryManager::finalize] Unable to deallocate pinned memory\n";
        //     msg += "FAKECUDA error - " + std::string(fakeCudaGetErrorName(cErr)) + "\n";
        //     msg += std::string(fakeCudaGetErrorString(cErr));
        //     throw std::runtime_error(msg);
        // }
        pinnedBuffer_ = nullptr;
        Logger::instance().log(  "[FakeCudaMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of pinned memory");
    }

    if (gpuBuffer_ != nullptr) {
      //fakeCudaError_t   cErr = fakeCudaFree(gpuBuffer_);
        free(gpuBuffer_);
        // if (cErr != fakeCudaSuccess) {
        //     std::string  msg = "[FakeCudaMemoryManager::finalize] Unable to deallocate GPU memory\n";
        //     msg += "FAKECUDA error - " + std::string(fakeCudaGetErrorName(cErr)) + "\n";
        //     msg += std::string(fakeCudaGetErrorString(cErr));
        //     throw std::runtime_error(msg);
        // }
        gpuBuffer_ = nullptr;
        Logger::instance().log(  "[FakeCudaMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of GPU memory");
    }
 
    pinnedOffset_ = 0;
    gpuOffset_ = 0;

    pthread_mutex_unlock(&mutex_);

    pthread_cond_destroy(&memoryReleased_);
    pthread_mutex_destroy(&mutex_);

    finalized_ = true;

    Logger::instance().log("[FakeCudaMemoryManager] Finalized");
}

/**
 *
 * \return 
 */
FakeCudaMemoryManager&   FakeCudaMemoryManager::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[FakeCudaMemoryManager::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[FakeCudaMemoryManager::instance] No access after finalization");
    }

    static FakeCudaMemoryManager   manager;
    return manager;
}

/**
 * 
 *
 */
FakeCudaMemoryManager::FakeCudaMemoryManager(void)
    : pinnedBuffer_{nullptr},
      gpuBuffer_{nullptr},
      pinnedOffset_{0},
      gpuOffset_{0}
{
    // TODO: How to get available RAM size in portable way?

    pthread_cond_init(&memoryReleased_, NULL);
    pthread_mutex_init(&mutex_, NULL);

    pthread_mutex_lock(&mutex_);

    // fakeCudaError_t    cErr = fakeCudaMallocHost(&pinnedBuffer_, nBytes_);
    pinnedBuffer_ = static_cast<char*>(malloc(nBytes_));
    if (pinnedBuffer_ == nullptr) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[FakeCudaMemoryManager::FakeCudaMemoryManager] ";
        errMsg += "Unable to allocate fake 'pinned' memory\n";
        errMsg += "FakeCuda error\n";
        perror(NULL);
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[FakeCudaMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of pinned memory");

    // cErr = fakeCudaMalloc(&gpuBuffer_, nBytes_);
    gpuBuffer_ = static_cast<char*>(malloc(nBytes_));
    if (gpuBuffer_ == nullptr) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[FakeCudaMemoryManager::FakeCudaMemoryManager] ";
        errMsg += "Unable to allocate GPU memory\n";
        errMsg += "FakeCuda error\n";
        perror(NULL);
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[FakeCudaMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of GPU memory");

    pthread_mutex_unlock(&mutex_);
}

/**
 * 
 *
 */
FakeCudaMemoryManager::~FakeCudaMemoryManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[FakeCudaMemoryManager::~FakeCudaMemoryManager] ERROR - Not finalized"
                  << std::endl;
    }
}

/**
 * 
 *
 */
void   FakeCudaMemoryManager::reset(void) {
    // There is no mechanism for now for releasing memory on a per request
    // basis.  This just resets the entire object so that it appears that 
    // no memory has been given out yet.
    pthread_mutex_lock(&mutex_);
    pinnedOffset_ = 0;
    gpuOffset_ = 0;
    pthread_mutex_unlock(&mutex_);

    Logger::instance().log("[FakeCudaMemoryManager] Reset memory resources");
}

/**
 * Refer to the documentation of the requestGpuMemory function of RuntimeBackend
 * for more information.
 *
 * @todo Confirm that the request is inline with byte alignment?
 */
void  FakeCudaMemoryManager::requestMemory(const std::size_t pinnedBytes,
                                       void** pinnedPtr,
                                       const std::size_t gpuBytes,
                                       void** gpuPtr) {
    if ((pinnedBytes == 0) || (gpuBytes == 0)) {
        std::string  errMsg = "[FakeCudaMemoryManager::requestMemory] ";
        errMsg += "Requests of zero indicate logical error\n";
        throw std::invalid_argument(errMsg);
    }

    pthread_mutex_lock(&mutex_);

    if ((pinnedOffset_ + pinnedBytes) > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[FakeCudaMemoryManager::requestMemory] ";
        errMsg += "Pinned buffer overflow\n";
        errMsg += std::to_string(nBytes_ - pinnedOffset_);
        errMsg += " bytes available and ";
        errMsg += std::to_string(pinnedBytes);
        errMsg += " bytes requested";
        throw std::overflow_error(errMsg);
    } else if ((gpuOffset_ + gpuBytes) > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[FakeCudaMemoryManager::requestMemory] ";
        errMsg += "GPU buffer overflow\n";
        errMsg += std::to_string(nBytes_ - gpuOffset_);
        errMsg += " bytes available and ";
        errMsg += std::to_string(gpuBytes);
        errMsg += " bytes requested";
        throw std::overflow_error(errMsg);
    }

    *pinnedPtr = static_cast<void*>(pinnedBuffer_ + pinnedOffset_);
    *gpuPtr    = static_cast<void*>(gpuBuffer_    + gpuOffset_);
    pinnedOffset_ += pinnedBytes;
    gpuOffset_    += gpuBytes;

    pthread_mutex_unlock(&mutex_);
}

/**
 * Refer to the documentation of the releaseGpuMemory function of RuntimeBackend
 * for more information.
 */
void  FakeCudaMemoryManager::releaseMemory(void** pinnedPtr, void** gpuPtr) {
    // Null so that we don't have dangling pointers.  This is inline with
    // the present reset() ugliness --- at the end of a runtime execution cycle,
    // all data packets should have called this routine so that effectively
    // none of the memory in the pools is checked out.
    *pinnedPtr = nullptr;
    *gpuPtr    = nullptr;
}

}

