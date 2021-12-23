#include "Milhoja_CudaBackend.h"

#include "Milhoja_CudaGpuEnvironment.h"
#include "Milhoja_CudaStreamManager.h"
#include "Milhoja_CudaMemoryManager.h"

namespace milhoja {

/**
 * Instantiate a CudaBackend object and all the helpers that it relies on.  It
 * is intended that this only ever be called by the RuntimeBackend's instantiate
 * member function.  In this way, the instatiated object should be a singleton.
 */
CudaBackend::CudaBackend(const unsigned int nStreams,
                         const std::size_t  nBytesInMemoryPools) {
    // Since RuntimeBackend calls instance() inside instantiate() and this constructor
    // should only be called once, these lines effectively carry out the 
    // instantiate() work of this derived class.
    CudaGpuEnvironment::instantiate();
    CudaStreamManager::instantiate(nStreams);
    CudaMemoryManager::instantiate(nBytesInMemoryPools);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  CudaBackend::maxNumberStreams(void) const {
    return CudaStreamManager::instance().maxNumberStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
int  CudaBackend::numberFreeStreams(void) {
    return CudaStreamManager::instance().numberFreeStreams();
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
Stream    CudaBackend::requestStream(const bool block) {
    return CudaStreamManager::instance().requestStream(block);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      CudaBackend::releaseStream(Stream& stream) {
    CudaStreamManager::instance().releaseStream(stream);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void  CudaBackend::initiateHostToGpuTransfer(DataPacket& packet) {
    // One and only one thread should own this packet at any given time.
    // Therefore, it has exclusive access and this code is thread-safe.
    cudaError_t cErr = cudaMemcpyAsync(packet.copyToGpuStart_gpu(),
                                       packet.copyToGpuStart_host(),
                                       packet.copyToGpuSizeInBytes(),
                                       cudaMemcpyHostToDevice,
                                       packet.stream());
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaBackend::initiateHostToGpuTransfer] ";
        errMsg += "Unable to initiate H-to-Gpu transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
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
 * \param  callback - the routine that will be registered with the CUDA runtime
 *                    so that the routine can unpack the packet (likely using
 *                    unpack) and perform other desired actions.
 * \param  callbackData - the data that must be passed to the callback so that
 *                        it can carry out its work.  This resource just passes
 *                        through this routine so that this routine has no
 *                        responsibility in managing the resources.
 */
void  CudaBackend::initiateGpuToHostTransfer(DataPacket& packet,
                                             GPU_TO_HOST_CALLBACK_FCN callback,
                                             void* callbackData) {
    // One and only one thread should own this packet at any given time.
    // Therefore, it has exclusive access and this code is thread-safe.
    cudaStream_t  stream = packet.stream();

    cudaError_t   cErr = cudaMemcpyAsync(packet.returnToHostStart_host(),
                                         packet.returnToHostStart_gpu(),
                                         packet.returnToHostSizeInBytes(),
                                         cudaMemcpyDeviceToHost,
                                         stream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaBackend::initiateGpuToHostTransfer] ";
        errMsg += "Unable to initiate Gpu-to-H transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    cErr = cudaLaunchHostFunc(stream, callback, callbackData); 
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaBackend::initiateGpuToHostTransfer] ";
        errMsg += "Unable to register Gpu-to-H callback function\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      CudaBackend::requestGpuMemory(const std::size_t pinnedBytes,
                                        void** pinnedPtr,
                                        const std::size_t gpuBytes,
                                        void** gpuPtr) {
    CudaMemoryManager::instance().requestMemory(pinnedBytes, pinnedPtr,
                                                gpuBytes, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      CudaBackend::releaseGpuMemory(void** pinnedPtr, void** gpuPtr) {
    CudaMemoryManager::instance().releaseMemory(pinnedPtr, gpuPtr);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 *
 */
void      CudaBackend::reset(void) {
    CudaMemoryManager::instance().reset();
}

}

