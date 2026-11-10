#include "Milhoja_HipMemoryManager.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include "Milhoja_HipGpuEnvironment.h"
#include "Milhoja_Logger.h"

namespace milhoja {

std::size_t HipMemoryManager::nBytes_ = 0;
bool HipMemoryManager::initialized_ = false;
bool HipMemoryManager::finalized_ = false;

namespace {

void checkHip(const hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string{"[HipMemoryManager::"} + operation + "] " +
            hipGetErrorName(error) + ": " + hipGetErrorString(error)
        );
    }
}

}

void HipMemoryManager::initialize(const std::size_t nBytesInMemoryPools) {
    if (initialized_) {
        throw std::logic_error("[HipMemoryManager::initialize] Already initialized");
    }
    if (nBytesInMemoryPools == 0) {
        throw std::invalid_argument("[HipMemoryManager::initialize] Empty pool");
    }
    nBytes_ = nBytesInMemoryPools;
    initialized_ = true;
    instance();
}

HipMemoryManager& HipMemoryManager::instance(void) {
    if (!initialized_) {
        throw std::logic_error("[HipMemoryManager::instance] Not initialized");
    }
    if (finalized_) {
        throw std::logic_error("[HipMemoryManager::instance] Already finalized");
    }
    static HipMemoryManager manager;
    return manager;
}

HipMemoryManager::HipMemoryManager(void)
    : pinnedBuffer_{nullptr},
      gpuBuffer_{nullptr},
      pinnedOffset_{0},
      gpuOffset_{0} {
    if (nBytes_ >= HipGpuEnvironment::instance().bytesInDeviceMemory()) {
        throw std::invalid_argument("[HipMemoryManager] Pool exceeds device memory");
    }
    pthread_mutex_init(&mutex_, nullptr);
    pthread_mutex_lock(&mutex_);
    try {
        checkHip(
            hipHostMalloc(reinterpret_cast<void**>(&pinnedBuffer_), nBytes_),
            "hipHostMalloc"
        );
        checkHip(
            hipMalloc(reinterpret_cast<void**>(&gpuBuffer_), nBytes_),
            "hipMalloc"
        );
    } catch (...) {
        if (pinnedBuffer_ != nullptr) {
            hipHostFree(pinnedBuffer_);
            pinnedBuffer_ = nullptr;
        }
        pthread_mutex_unlock(&mutex_);
        throw;
    }
    pthread_mutex_unlock(&mutex_);
    Logger::instance().log(
        "[HipMemoryManager] Allocated " + std::to_string(nBytes_) +
        " bytes in each memory pool"
    );
}

void HipMemoryManager::finalize(void) {
    if (!initialized_ || finalized_) {
        throw std::logic_error("[HipMemoryManager::finalize] Invalid state");
    }
    pthread_mutex_lock(&mutex_);
    if (gpuBuffer_ != nullptr) {
        checkHip(hipFree(gpuBuffer_), "hipFree");
        gpuBuffer_ = nullptr;
    }
    if (pinnedBuffer_ != nullptr) {
        checkHip(hipHostFree(pinnedBuffer_), "hipHostFree");
        pinnedBuffer_ = nullptr;
    }
    pinnedOffset_ = 0;
    gpuOffset_ = 0;
    pthread_mutex_unlock(&mutex_);
    pthread_mutex_destroy(&mutex_);
    finalized_ = true;
}

void HipMemoryManager::requestMemory(const std::size_t pinnedBytes,
                                     void** pinnedPtr,
                                     const std::size_t gpuBytes,
                                     void** gpuPtr) {
    if (pinnedBytes == 0 || gpuBytes == 0 || pinnedPtr == nullptr || gpuPtr == nullptr) {
        throw std::invalid_argument("[HipMemoryManager::requestMemory] Invalid request");
    }
    pthread_mutex_lock(&mutex_);
    if (pinnedOffset_ + pinnedBytes > nBytes_ || gpuOffset_ + gpuBytes > nBytes_) {
        pthread_mutex_unlock(&mutex_);
        throw std::overflow_error("[HipMemoryManager::requestMemory] Pool overflow");
    }
    *pinnedPtr = pinnedBuffer_ + pinnedOffset_;
    *gpuPtr = gpuBuffer_ + gpuOffset_;
    pinnedOffset_ += pinnedBytes;
    gpuOffset_ += gpuBytes;
    pthread_mutex_unlock(&mutex_);
}

void HipMemoryManager::releaseMemory(void** pinnedPtr, void** gpuPtr) {
    if (pinnedPtr == nullptr || gpuPtr == nullptr) {
        throw std::invalid_argument("[HipMemoryManager::releaseMemory] Null handle");
    }
    *pinnedPtr = nullptr;
    *gpuPtr = nullptr;
}

void HipMemoryManager::reset(void) {
    pthread_mutex_lock(&mutex_);
    pinnedOffset_ = 0;
    gpuOffset_ = 0;
    pthread_mutex_unlock(&mutex_);
}

HipMemoryManager::~HipMemoryManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[HipMemoryManager::~HipMemoryManager] ERROR - Not finalized\n";
    }
}

}
