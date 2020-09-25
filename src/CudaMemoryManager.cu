#include "CudaMemoryManager.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

#include "OrchestrationLogger.h"
#include "CudaGpuEnvironment.h"

namespace orchestration {

// Default value chosen in conjunction with the error checking in the
// constructor such that client code will get an error if they do not explicitly
// set the numer of bytes before accessing the manager.
std::size_t   CudaMemoryManager::nBytes_ = 0;
bool          CudaMemoryManager::wasInstantiated_ = false;

/**
 * Before calling this routine, client code must first set the size of memory
 * buffers to be managed using setBufferSize().
 *
 * \return 
 */
CudaMemoryManager&   CudaMemoryManager::instance(void) {
    static CudaMemoryManager   manager;
    return manager;
}

/**
 * This member must be called before accessing the manager, but cannot be called
 * after accessing the manager.
 *
 * \return 
 */
void CudaMemoryManager::setBufferSize(const std::size_t bytes) {
    if (wasInstantiated_) {
        throw std::logic_error("[CudaMemoryManager::setBufferSize] "
                               "Cannot be set once the manager has been accessed");
    } else if (bytes == 0) {
        throw std::invalid_argument("[CudaMemoryManager::setBufferSize] "
                                    "Buffers must be non-empty");
    }
    // TODO: Check that buffers are sized for byte alignment?

    nBytes_ = bytes;
    Logger::instance().log( "[CudaMemoryManager] Buffer size set to "
                           + std::to_string(nBytes_ / std::pow(1024.0, 3.0))
                           + " Gb");
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
    Logger::instance().log("[CudaMemoryManager] Initializing...");

    std::size_t   gpuMemBytes = CudaGpuEnvironment::instance().bytesInDeviceMemory();
    Logger::instance().log(  "[CudaMemoryManager] GPU memory has " 
                           + std::to_string(gpuMemBytes / std::pow(1024.0, 3.0))
                           + " Gb");
    // TODO: How to get RAM size in portable way?

    if (nBytes_ == 0) {
        throw std::invalid_argument("[CudaMemoryManager::CudaMemoryManager] "
                                    "Set buffer size before accessing manager");
    } else if (nBytes_ >= gpuMemBytes) {
        throw std::invalid_argument("[CudaMemoryManager::CudaMemoryManager] "
                                    "Cannot use all GPU memory as buffer");
    }

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

    wasInstantiated_ = true;
    pthread_mutex_unlock(&mutex_);

    Logger::instance().log("[CudaMemoryManager] Created and ready for use");
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
    wasInstantiated_ = false;

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

