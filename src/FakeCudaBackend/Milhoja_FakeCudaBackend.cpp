#include "Milhoja_FakeCudaBackend.h"

#include "Milhoja_Logger.h"
#include "Milhoja_FakeCudaGpuEnvironment.h"
#include "Milhoja_FakeCudaStreamManager.h"
#include "Milhoja_FakeCudaMemoryManager.h"

namespace milhoja {

/**
 * Initialize a FakeCudaBackend object and all the helpers that it relies on.  It
 * is intended that this only ever be called by the RuntimeBackend's initialize
 * member function.  In this way, the initialized object should be a singleton.
 */
FakeCudaBackend::FakeCudaBackend(const unsigned int nStreams,
                         const std::size_t  nBytesInMemoryPools) {
    Logger::instance().log("[FakeCudaBackend] Initializing...");

    // Since RuntimeBackend calls instance() inside initialize() and this constructor
    // should only be called once, these lines effectively carry out the 
    // initialize() work of this derived class.
    FakeCudaGpuEnvironment::initialize();
    FakeCudaStreamManager::initialize(nStreams);
    FakeCudaMemoryManager::initialize(nBytesInMemoryPools);

    Logger::instance().log("[FakeCudaBackend] Created and ready for use");
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void    FakeCudaBackend::finalize(void) {
    Logger::instance().log("[FakeCudaBackend] Finalizing...");

    FakeCudaMemoryManager::instance().finalize();
    FakeCudaStreamManager::instance().finalize();
    FakeCudaGpuEnvironment::instance().finalize();

    RuntimeBackend::finalize();

    Logger::instance().log("[FakeCudaBackend] Finalized");
}


/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  FakeCudaBackend::maxNumberStreams(void) const {
    return FakeCudaStreamManager::instance().maxNumberStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  FakeCudaBackend::numberFreeStreams(void) {
    return FakeCudaStreamManager::instance().numberFreeStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
Stream    FakeCudaBackend::requestStream(const bool block) {
    return FakeCudaStreamManager::instance().requestStream(block);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      FakeCudaBackend::releaseStream(Stream& stream) {
    FakeCudaStreamManager::instance().releaseStream(stream);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void  FakeCudaBackend::initiateHostToGpuTransfer(DataPacket& packet) {
    // One and only one thread should own this packet at any given time.
    // Therefore, it has exclusive access and this code is thread-safe.
    // cudaError_t cErr = cudaMemcpyAsync(packet.copyToGpuStart_gpu(),
    //                                    packet.copyToGpuStart_host(),
    //                                    packet.copyToGpuSizeInBytes(),
    //                                    cudaMemcpyHostToDevice,
    //                                    packet.stream());
    // if (cErr != cudaSuccess) {
    //     std::string  errMsg = "[FakeCudaBackend::initiateHostToGpuTransfer] ";
    //     errMsg += "Unable to initiate H-to-Gpu transfer\n";
    //     errMsg += "FAKECUDA error\n";
    //     throw std::runtime_error(errMsg);
    // }
    memcpy(packet.copyToGpuStart_gpu(),
	   packet.copyToGpuStart_host(),
	   packet.copyToGpuSizeInBytes());
}

/**
 *  Refer to the RuntimeBackend documentation for general information.
 *
 *  Initiate an asychronous transfer of the packet from the device to the host
 *  on the packet's stream.  As part of this, launch on the same stream the given
 *  callback for handling the unpacking and other auxiliary work that must occur
 *  once the packet is back in pinned memory.
 *
 * \param  packet   - the data packet to transfer.
 * \param  callback - the routine that will be registered with the FAKECUDA runtime
 *                    so that the routine can unpack the packet (likely using
 *                    unpack) and perform other desired actions.
 * \param  callbackData - the data that must be passed to the callback so that
 *                        it can carry out its work.  This resource just passes
 *                        through this routine so that this routine has no
 *                        responsibility in managing the resources.
 */
void  FakeCudaBackend::initiateGpuToHostTransfer(DataPacket& packet,
                                             GPU_TO_HOST_CALLBACK_FCN callback,
                                             void* callbackData) {
    // One and only one thread should own this packet at any given time.
    // Therefore, it has exclusive access and this code is thread-safe.
    // cudaStream_t  stream = packet.stream();

    // cudaError_t   cErr = cudaMemcpyAsync(packet.returnToHostStart_host(),
    //                                      packet.returnToHostStart_gpu(),
    //                                      packet.returnToHostSizeInBytes(),
    //                                      cudaMemcpyDeviceToHost,
    //                                      stream);
    // if (cErr != cudaSuccess) {
    //     std::string  errMsg = "[FakeCudaBackend::initiateGpuToHostTransfer] ";
    //     errMsg += "Unable to initiate Gpu-to-H transfer\n";
    //     errMsg += "FAKECUDA error\n";
    //     throw std::runtime_error(errMsg);
    // }
    memcpy(packet.returnToHostStart_host(),
	   packet.returnToHostStart_gpu(),
	   packet.returnToHostSizeInBytes());

    // cErr = cudaLaunchHostFunc(stream, callback, callbackData); 
    // if (cErr != cudaSuccess) {
    //     std::string  errMsg = "[FakeCudaBackend::initiateGpuToHostTransfer] ";
    //     errMsg += "Unable to register Gpu-to-H callback function\n";
    //     errMsg += "FAKECUDA error\n";
    //     throw std::runtime_error(errMsg);
    // }
    callback(callbackData);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      FakeCudaBackend::requestGpuMemory(const std::size_t pinnedBytes,
                                        void** pinnedPtr,
                                        const std::size_t gpuBytes,
                                        void** gpuPtr) {
    FakeCudaMemoryManager::instance().requestMemory(pinnedBytes, pinnedPtr,
                                                gpuBytes, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      FakeCudaBackend::releaseGpuMemory(void** pinnedPtr, void** gpuPtr) {
    FakeCudaMemoryManager::instance().releaseMemory(pinnedPtr, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 *
 */
void      FakeCudaBackend::reset(void) {
    FakeCudaMemoryManager::instance().reset();
}

}

