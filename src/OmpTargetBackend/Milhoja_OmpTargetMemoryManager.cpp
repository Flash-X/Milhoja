#include <stdexcept>
#include "Milhoja_OmpTargetMemoryManager.h"

#include <cmath>
#include <iostream>

#include "Milhoja_Logger.h"
#include "Milhoja_OmpTargetGpuEnvironment.h"
#include "Milhoja_OmpTargetBackend.h"

namespace milhoja {

std::size_t   OmpTargetMemoryManager::nBytes_ = 0;
bool          OmpTargetMemoryManager::initialized_ = false;
bool          OmpTargetMemoryManager::finalized_ = false;
int           OmpTargetMemoryManager::device_num_ = -1;
omp_allocator_handle_t OmpTargetMemoryManager::pinned_allocator_ = omp_null_allocator;

/**
 *
 */
  void OmpTargetMemoryManager::initialize(const std::size_t nBytesInMemoryPools,
                                          const int target_device_num) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    constexpr int N_PINNEDTRAITS = 2;
    if (initialized_) {
        throw std::logic_error("[OmpTargetMemoryManager::initialize] "
                               "Memory manager already initialized");
    } else if (nBytesInMemoryPools == 0) {
        throw std::invalid_argument("[OmpTargetMemoryManager::initialize] "
                                    "Buffers must be non-empty");
    }
    // TODO: Check that buffers are sized for byte alignment?

    Logger::instance().log("[OmpTargetMemoryManager] Initializing...");

    nBytes_ = nBytesInMemoryPools;
    device_num_ = target_device_num;

    omp_memspace_handle_t pinned_mem_space = omp_default_mem_space; // or a specific device memory space?

    omp_alloctrait_t traits[N_PINNEDTRAITS];
    traits[0].key = omp_atk_alignment;
    traits[0].value = 4096; // alignment for pinned memory
    traits[1].key = omp_atk_pinned;
    traits[1].value = omp_atv_true; // Request pinned memory

    Logger::instance().log("[OmpTargetMemoryManager] Requesting pinned allocator with "
			   + std::to_string(N_PINNEDTRAITS)
			   + " traits...");
    pinned_allocator_ = omp_init_allocator(pinned_mem_space, 1, traits);
    initialized_ = true;

    instance();

    Logger::instance().log("[OmpTargetMemoryManager] Created and ready for use");
}

/**
 *
 */
void    OmpTargetMemoryManager::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[OmpTargetMemoryManager::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[OmpTargetMemoryManager::finalize] Already finalized");
    }

    Logger::instance().log("[OmpTargetMemoryManager] Finalizing ...");

    pthread_mutex_lock(&mutex_);

    if (pinnedBuffer_ != nullptr) {
      //cudaError_t   cErr = cudaFreeHost(pinnedBuffer_);
      omp_free(pinnedBuffer_, pinned_allocator_);
	// if (cErr != cudaSuccess) {
        //    std::string  msg = "[OmpTargetMemoryManager::finalize] Unable to deallocate pinned memory\n";
        //    msg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        //    msg += std::string(cudaGetErrorString(cErr));
        //    throw std::runtime_error(msg);
        // }
        pinnedBuffer_ = nullptr;
        Logger::instance().log(  "[OmpTargetMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of pinned memory");
    }
    omp_destroy_allocator(pinned_allocator_);
    Logger::instance().log("[OmpTargetMemoryManager] Destroyed pinned_allocator_");

    if (gpuBuffer_ != nullptr) {
      //cudaError_t   cErr = cudaFree(gpuBuffer_);
      omp_target_free(gpuBuffer_, device_num_);
        // if (cErr != cudaSuccess) {
        //     std::string  msg = "[OmpTargetMemoryManager::finalize] Unable to deallocate GPU memory\n";
        //     msg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        //     msg += std::string(cudaGetErrorString(cErr));
        //     throw std::runtime_error(msg);
        // }
        gpuBuffer_ = nullptr;
        Logger::instance().log(  "[OmpTargetMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of GPU memory");
    }
 
    pinnedOffset_ = 0;
    gpuOffset_ = 0;

    pthread_mutex_unlock(&mutex_);

    pthread_cond_destroy(&memoryReleased_);
    pthread_mutex_destroy(&mutex_);

    finalized_ = true;

    Logger::instance().log("[OmpTargetMemoryManager] Finalized");
}

/**
 *
 * \return 
 */
OmpTargetMemoryManager&   OmpTargetMemoryManager::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[OmpTargetMemoryManager::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[OmpTargetMemoryManager::instance] No access after finalization");
    }

    static OmpTargetMemoryManager   manager;
    return manager;
}

/**
 * 
 *
 */
OmpTargetMemoryManager::OmpTargetMemoryManager(void)
    : pinnedBuffer_{nullptr},
      gpuBuffer_{nullptr},
      pinnedOffset_{0},
      gpuOffset_{0}
{
    std::size_t   gpuMemBytes = OmpTargetGpuEnvironment::instance().bytesInDeviceMemory();
    Logger::instance().log(  "[OmpTargetMemoryManager] GPU memory has " 
                           + std::to_string(gpuMemBytes / std::pow(1024.0, 3.0))
                           + " Gb");
    if (nBytes_ >= gpuMemBytes) {
        throw std::invalid_argument("[OmpTargetMemoryManager::OmpTargetMemoryManager] "
                                    "Cannot use all GPU memory as buffer");
    }
    // TODO: How to get RAM size in portable way?

    pthread_cond_init(&memoryReleased_, NULL);
    pthread_mutex_init(&mutex_, NULL);

    pthread_mutex_lock(&mutex_);
    Logger::instance().log(  "[OmpTargetMemoryManager] DBG mutex_ locked" );

    // cudaError_t    cErr = cudaMallocHost(&pinnedBuffer_, nBytes_);
    pinnedBuffer_ = static_cast<char*>(omp_alloc(nBytes_, pinned_allocator_));
    Logger::instance().log(  "[OmpTargetMemoryManager] DBG pinned allocated" );
    if (pinnedBuffer_ == nullptr) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[OmpTargetMemoryManager::OmpTargetMemoryManager] ";
        errMsg += "Unable to allocate pinned memory\n";
        errMsg += "OmpTarget error\n";
        perror(NULL);
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[OmpTargetMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of pinned memory");

    // cErr = cudaMalloc(&gpuBuffer_, nBytes_);
    gpuBuffer_ = static_cast<char*>(omp_target_alloc(nBytes_, device_num_));
    if (gpuBuffer_ == nullptr) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[OmpTargetMemoryManager::OmpTargetMemoryManager] ";
        errMsg += "Unable to allocate OpenMP target device memory\n";
        errMsg += "OmpTarget error\n";
        perror(NULL);
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[OmpTargetMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of GPU memory");

    pthread_mutex_unlock(&mutex_);
}

/**
 * 
 *
 */
OmpTargetMemoryManager::~OmpTargetMemoryManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[OmpTargetMemoryManager::~OmpTargetMemoryManager] ERROR - Not finalized"
                  << std::endl;
    }
}

/**
 * 
 *
 */
void   OmpTargetMemoryManager::reset(void) {
    // There is no mechanism for now for releasing memory on a per request
    // basis.  This just resets the entire object so that it appears that 
    // no memory has been given out yet.
    pthread_mutex_lock(&mutex_);
    pinnedOffset_ = 0;
    gpuOffset_ = 0;
    pthread_mutex_unlock(&mutex_);

    Logger::instance().log("[OmpTargetMemoryManager] Reset memory resources");
}

/**
 * Refer to the documentation of the requestGpuMemory function of RuntimeBackend
 * for more information.
 *
 * @todo Confirm that the request is inline with byte alignment?
 */
void  OmpTargetMemoryManager::requestMemory(const std::size_t pinnedBytes,
                                       void** pinnedPtr,
                                       const std::size_t gpuBytes,
                                       void** gpuPtr) {
    if ((pinnedBytes == 0) || (gpuBytes == 0)) {
        std::string  errMsg = "[OmpTargetMemoryManager::requestMemory] ";
        errMsg += "Requests of zero indicate logical error\n";
        throw std::invalid_argument(errMsg);
    }

    pthread_mutex_lock(&mutex_);

    if ((pinnedOffset_ + pinnedBytes) > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[OmpTargetMemoryManager::requestMemory] ";
        errMsg += "Pinned buffer overflow\n";
        errMsg += std::to_string(nBytes_ - pinnedOffset_);
        errMsg += " bytes available and ";
        errMsg += std::to_string(pinnedBytes);
        errMsg += " bytes requested";
        throw std::overflow_error(errMsg);
    } else if ((gpuOffset_ + gpuBytes) > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[OmpTargetMemoryManager::requestMemory] ";
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
void  OmpTargetMemoryManager::releaseMemory(void** pinnedPtr, void** gpuPtr) {
    // Null so that we don't have dangling pointers.  This is inline with
    // the present reset() ugliness --- at the end of a runtime execution cycle,
    // all data packets should have called this routine so that effectively
    // none of the memory in the pools is checked out.
    *pinnedPtr = nullptr;
    *gpuPtr    = nullptr;
}

}

