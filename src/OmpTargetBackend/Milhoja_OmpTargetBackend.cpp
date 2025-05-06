#include "Milhoja_OmpTargetBackend.h"

#include "Milhoja_Logger.h"
#include "Milhoja_OmpTargetGpuEnvironment.h"
#include "Milhoja_OmpTargetStreamManager.h"
#include "Milhoja_OmpTargetMemoryManager.h"

namespace milhoja {

/**
 * Initialize an OmpTargetBackend object and all the helpers that it relies on.
 * It is intended that this only ever be called by the RuntimeBackend's initialize
 * member function.  In this way, the initialized object should be a singleton.
 */
OmpTargetBackend::OmpTargetBackend(const unsigned int nStreams,
                         const std::size_t  nBytesInMemoryPools)
  : target_device_num_{omp_get_default_device()}
{
    Logger::instance().log("[OmpTargetBackend] Initializing...");

    // Since RuntimeBackend calls instance() inside initialize() and this constructor
    // should only be called once, these lines effectively carry out the 
    // initialize() work of this derived class.
    OmpTargetGpuEnvironment::initialize();
    OmpTargetStreamManager::initialize(nStreams);
    OmpTargetMemoryManager::initialize(nBytesInMemoryPools, target_device_num_);

    Logger::instance().log("[OmpTargetBackend] Created and ready for use");
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void    OmpTargetBackend::finalize(void) {
    Logger::instance().log("[OmpTargetBackend] Finalizing...");

    OmpTargetMemoryManager::instance().finalize();
    OmpTargetStreamManager::instance().finalize();
    OmpTargetGpuEnvironment::instance().finalize();

    RuntimeBackend::finalize();

    Logger::instance().log("[OmpTargetBackend] Finalized");
}


/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  OmpTargetBackend::maxNumberStreams(void) const {
    return OmpTargetStreamManager::instance().maxNumberStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  OmpTargetBackend::numberFreeStreams(void) {
    return OmpTargetStreamManager::instance().numberFreeStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
Stream    OmpTargetBackend::requestStream(const bool block) {
    return OmpTargetStreamManager::instance().requestStream(block);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      OmpTargetBackend::releaseStream(Stream& stream) {
    OmpTargetStreamManager::instance().releaseStream(stream);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void  OmpTargetBackend::initiateHostToGpuTransfer(DataPacket& packet) {
    // One and only one thread should own this packet at any given time.
    // Therefore, it has exclusive access and this code is thread-safe.
    // cudaError_t cErr = cudaMemcpyAsync(packet.copyToGpuStart_gpu(),
    //                                    packet.copyToGpuStart_host(),
    //                                    packet.copyToGpuSizeInBytes(),
    //                                    cudaMemcpyHostToDevice,
    //                                    packet.stream());
    int rv = omp_target_memcpy(packet.copyToGpuStart_gpu(),
			       packet.copyToGpuStart_host(),
			       packet.copyToGpuSizeInBytes(), 0, 0,
			       target_device_num_,
			       omp_get_initial_device());
    if (rv != 0) {
        std::string  errMsg = "[OmpTargetBackend::initiateHostToGpuTransfer] ";
        errMsg += "Unable to run H-to-Gpu transfer\n";
        errMsg += "omp_target_memcpy error\n";
        throw std::runtime_error(errMsg);
    }
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
 * \param  callback - the routine that will be registered with the CUDA runtime
 *                    so that the routine can unpack the packet (likely using
 *                    unpack) and perform other desired actions.
 * \param  callbackData - the data that must be passed to the callback so that
 *                        it can carry out its work.  This resource just passes
 *                        through this routine so that this routine has no
 *                        responsibility in managing the resources.
 */
void  OmpTargetBackend::initiateGpuToHostTransfer(DataPacket& packet,
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
    int rv = omp_target_memcpy(packet.returnToHostStart_host(),
			       packet.returnToHostStart_gpu(),
			       packet.returnToHostSizeInBytes(), 0, 0,
			       omp_get_initial_device(),
			       target_device_num_);
    if (rv != 0) {
        std::string  errMsg = "[OmpTargetBackend::initiateGpuToHostTransfer] ";
        errMsg += "Unable to initiate Gpu-to-H transfer\n";
        errMsg += "omp_target_memcpy error\n";
        throw std::runtime_error(errMsg);
    }

    // cErr = cudaLaunchHostFunc(stream, callback, callbackData); 
    // if (cErr != cudaSuccess) {
    //     std::string  errMsg = "[OmpTargetBackend::initiateGpuToHostTransfer] ";
    //     errMsg += "Unable to register Gpu-to-H callback function\n";
    //     errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
    //     errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
    //     throw std::runtime_error(errMsg);
    // }
    callback(callbackData);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      OmpTargetBackend::requestGpuMemory(const std::size_t pinnedBytes,
                                        void** pinnedPtr,
                                        const std::size_t gpuBytes,
                                        void** gpuPtr) {
    OmpTargetMemoryManager::instance().requestMemory(pinnedBytes, pinnedPtr,
                                                gpuBytes, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      OmpTargetBackend::releaseGpuMemory(void** pinnedPtr, void** gpuPtr) {
    OmpTargetMemoryManager::instance().releaseMemory(pinnedPtr, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 *
 */
void      OmpTargetBackend::reset(void) {
    OmpTargetMemoryManager::instance().reset();
}

}

