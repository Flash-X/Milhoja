#ifndef USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

#include "CudaMemoryManager.h"

#include <stdexcept>
#include <iostream>

#include "OrchestrationLogger.h"
#include "CudaGpuEnvironment.h"

namespace orchestration {

std::size_t   CudaMemoryManager::nBytes_ = 0;
bool          CudaMemoryManager::instantiated_ = false;

/**
 *
 * \return 
 */
void CudaMemoryManager::instantiate(const std::size_t nBytesInMemoryPools) {
    Logger::instance().log("[CudaMemoryManager] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[CudaMemoryManager::instantiate] "
                               "StreamManager already instantiated");
    } else if (nBytesInMemoryPools == 0) {
        throw std::invalid_argument("[CudaMemoryManager::instantiate] "
                                    "Buffers must be non-empty");
    }
    // TODO: Check that buffers are sized for byte alignment?

    nBytes_ = nBytesInMemoryPools;
    instantiated_ = true;

    instance();

    Logger::instance().log("[CudaMemoryManager] Created and ready for use");
}

/**
 *
 * \return 
 */
CudaMemoryManager&   CudaMemoryManager::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[CudaMemoryManager::instance] Instantiate first");
    }

    static CudaMemoryManager   manager;
    return manager;
}

/**
 * 
 *
 * \return 
 */
CudaMemoryManager::CudaMemoryManager(void)
    : pinnedBuffer_{nullptr},
      gpuBuffer_{nullptr},
      offset_{0}
{
    std::size_t   gpuMemBytes = CudaGpuEnvironment::instance().bytesInDeviceMemory();
    Logger::instance().log(  "[CudaMemoryManager] GPU memory has " 
                           + std::to_string(gpuMemBytes / std::pow(1024.0, 3.0))
                           + " Gb");
    if (nBytes_ >= gpuMemBytes) {
        throw std::invalid_argument("[CudaMemoryManager::CudaMemoryManager] "
                                    "Cannot use all GPU memory as buffer");
    }
    // TODO: How to get RAM size in portable way?

    pthread_cond_init(&memoryReleased_, NULL);
    pthread_mutex_init(&mutex_, NULL);

    pthread_mutex_lock(&mutex_);

    cudaError_t    cErr = cudaMallocHost(&pinnedBuffer_, nBytes_);
    if ((cErr != cudaSuccess) || (pinnedBuffer_ == nullptr)) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[CudaMemoryManager::CudaMemoryManager] ";
        errMsg += "Unable to allocate pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[CudaMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of pinned memory");

    cErr = cudaMalloc(&gpuBuffer_, nBytes_);
    if ((cErr != cudaSuccess) || (gpuBuffer_ == nullptr)) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[CudaMemoryManager::CudaMemoryManager] ";
        errMsg += "Unable to allocate GPU memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    Logger::instance().log(  "[CudaMemoryManager] Allocated " 
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb of GPU memory");

    pthread_mutex_unlock(&mutex_);
}

/**
 * 
 *
 * \return 
 */
CudaMemoryManager::~CudaMemoryManager(void) {
    Logger::instance().log("[CudaMemoryManager] Finalizing...");

    pthread_mutex_lock(&mutex_);

    if (pinnedBuffer_ != nullptr) {
        cudaError_t   cErr = cudaFreeHost(pinnedBuffer_);
        if (cErr != cudaSuccess) {
            std::cerr << "[CudaMemoryManager::~CudaMemoryManager] "
                      << "Unable to deallocate pinned memory\n"
                      << "CUDA error - "
                      << cudaGetErrorName(cErr) << "\n"
                      << cudaGetErrorString(cErr) << std::endl;
            pthread_mutex_unlock(&mutex_);
        }
        pinnedBuffer_ = nullptr;
        Logger::instance().log(  "[CudaMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of pinned memory");
    }

    if (gpuBuffer_ != nullptr) {
        cudaError_t   cErr = cudaFree(gpuBuffer_);
        if (cErr != cudaSuccess) {
            std::cerr << "[CudaMemoryManager::~CudaMemoryManager] "
                      << "Unable to deallocate GPU memory\n"
                      << "CUDA error - "
                      << cudaGetErrorName(cErr) << "\n"
                      << cudaGetErrorString(cErr) << std::endl;
            pthread_mutex_unlock(&mutex_);
        }
        gpuBuffer_ = nullptr;
        Logger::instance().log(  "[CudaMemoryManager] Deallocated " 
                               + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                               + " Gb of GPU memory");
    }
 
    offset_ = 0;
    instantiated_ = false;

    pthread_mutex_unlock(&mutex_);

    pthread_cond_destroy(&memoryReleased_);
    pthread_mutex_destroy(&mutex_);

    Logger::instance().log("[CudaMemoryManager] Destroyed");
}

/**
 * 
 *
 * \return 
 */
void   CudaMemoryManager::reset(void) {
    // There is no mechanism for now for releasing memory on a per request
    // basis.  This just resets the entire object so that it appears that 
    // no memory has been given out yet.
    pthread_mutex_lock(&mutex_);
    offset_ = 0;
    pthread_mutex_unlock(&mutex_);

    Logger::instance().log("[CudaMemoryManager] Reset memory resources");
}

/**
 * 
 *
 * \return 
 */
void  CudaMemoryManager::requestMemory(const std::size_t bytes, void** hostPtr, void** gpuPtr) {
    if (bytes == 0) {
        std::string  errMsg = "[CudaMemoryManager::requestMemory] ";
        errMsg += "Requests of zero indicate logical error\n";
        throw std::invalid_argument(errMsg);
    }

    pthread_mutex_lock(&mutex_);

    // TODO: Something like this should block.
    if ((offset_ + bytes) > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        std::string  errMsg = "[CudaMemoryManager::requestMemory] ";
        errMsg += "Buffer overflow\n";
        errMsg += std::to_string(nBytes_ - offset_);
        errMsg += " bytes available and ";
        errMsg += std::to_string(bytes);
        errMsg += " bytes requested";
        throw std::overflow_error(errMsg);
    }
    // TODO: Confirm that the request is inline with byte alignment?

    *hostPtr = static_cast<void*>(pinnedBuffer_ + offset_);
    *gpuPtr  = static_cast<void*>(gpuBuffer_    + offset_);
    offset_ += bytes;

    pthread_mutex_unlock(&mutex_);
}

/**
 * 
 *
 * \return 
 */
void  CudaMemoryManager::releaseMemory(void** hostPtr, void** gpuPtr) {
    // Memory requests are permanent for now (i.e. we leak if this gets called).
    *hostPtr = nullptr;
    *gpuPtr  = nullptr;
}

}

