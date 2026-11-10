#include "Milhoja_HipBackend.h"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include "Milhoja_HipGpuEnvironment.h"
#include "Milhoja_HipMemoryManager.h"
#include "Milhoja_HipStreamManager.h"
#include "Milhoja_Logger.h"

namespace milhoja {

namespace {

void checkHip(const hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string{"[HipBackend::"} + operation + "] " +
            hipGetErrorName(error) + ": " + hipGetErrorString(error)
        );
    }
}

}

HipBackend::HipBackend(const unsigned int nStreams,
                       const std::size_t nBytesInMemoryPools) {
    Logger::instance().log("[HipBackend] Initializing...");
    HipGpuEnvironment::initialize();
    HipStreamManager::initialize(nStreams);
    HipMemoryManager::initialize(nBytesInMemoryPools);
    Logger::instance().log("[HipBackend] Created and ready for use");
}

void HipBackend::finalize(void) {
    Logger::instance().log("[HipBackend] Finalizing...");
    HipMemoryManager::instance().finalize();
    HipStreamManager::instance().finalize();
    HipGpuEnvironment::instance().finalize();
    RuntimeBackend::finalize();
    Logger::instance().log("[HipBackend] Finalized");
}

int HipBackend::maxNumberStreams(void) const {
    return HipStreamManager::instance().maxNumberStreams();
}

int HipBackend::numberFreeStreams(void) {
    return HipStreamManager::instance().numberFreeStreams();
}

Stream HipBackend::requestStream(const bool block) {
    return HipStreamManager::instance().requestStream(block);
}

void HipBackend::releaseStream(Stream& stream) {
    HipStreamManager::instance().releaseStream(stream);
}

void HipBackend::synchronizeStream(const Stream& stream) {
    if (!stream.isValid()) {
        throw std::invalid_argument("[HipBackend::synchronizeStream] Invalid stream");
    }
    checkHip(hipStreamSynchronize(stream.hipStream), "synchronizeStream");
}

void HipBackend::initiateHostToGpuTransfer(DataPacket& packet) {
    checkHip(
        hipMemcpyAsync(packet.copyToGpuStart_gpu(),
                       packet.copyToGpuStart_host(),
                       packet.copyToGpuSizeInBytes(),
                       hipMemcpyHostToDevice,
                       packet.stream()),
        "initiateHostToGpuTransfer"
    );
}

void HipBackend::initiateGpuToHostTransfer(
    DataPacket& packet,
    GPU_TO_HOST_CALLBACK_FCN callback,
    void* callbackData
) {
    checkHip(
        hipMemcpyAsync(packet.returnToHostStart_host(),
                       packet.returnToHostStart_gpu(),
                       packet.returnToHostSizeInBytes(),
                       hipMemcpyDeviceToHost,
                       packet.stream()),
        "initiateGpuToHostTransfer"
    );
    checkHip(
        hipLaunchHostFunc(packet.stream(), callback, callbackData),
        "initiateGpuToHostTransfer callback"
    );
}

void HipBackend::requestGpuMemory(const std::size_t pinnedBytes,
                                  void** pinnedPtr,
                                  const std::size_t gpuBytes,
                                  void** gpuPtr) {
    HipMemoryManager::instance().requestMemory(
        pinnedBytes, pinnedPtr, gpuBytes, gpuPtr
    );
}

void HipBackend::releaseGpuMemory(void** pinnedPtr, void** gpuPtr) {
    HipMemoryManager::instance().releaseMemory(pinnedPtr, gpuPtr);
}

void HipBackend::reset(void) {
    HipMemoryManager::instance().reset();
}

}
