#ifndef USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

#include "CudaBackend.h"

#include "CudaGpuEnvironment.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

namespace orchestration {

/**
 *
 */
CudaBackend::CudaBackend(const unsigned int nStreams,
                         const std::size_t  nBytesInMemoryPools) {
    // Since Backend calls instance() inside instantiate() and this constructor
    // should only be called once, these lines effectively carry out the 
    // instantiation work of this derived class.
    CudaGpuEnvironment::instantiate();
    CudaStreamManager::instantiate(nStreams);
    CudaMemoryManager::instantiate(nBytesInMemoryPools);
}

/**
 *
 */
int  CudaBackend::maxNumberStreams(void) const {
    return CudaStreamManager::instance().maxNumberStreams();
}

/**
 *
 */
int  CudaBackend::numberFreeStreams(void) {
    return CudaStreamManager::instance().numberFreeStreams();
}

/**
 *
 */
Stream    CudaBackend::requestStream(const bool block) {
    return CudaStreamManager::instance().requestStream(block);
}

/**
 *
 */
void      CudaBackend::releaseStream(Stream& stream) {
    CudaStreamManager::instance().releaseStream(stream);
}

/**
 *
 */
void  CudaBackend::initiateHostToGpuTransfer(DataPacket& packet) {
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
                                             cudaHostFn_t callback,
                                             void* callbackData) {
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
 *
 */
void      CudaBackend::requestGpuMemory(const std::size_t pinnedBytes,
                                        void** pinnedPtr,
                                        const std::size_t gpuBytes,
                                        void** gpuPtr) {
    CudaMemoryManager::instance().requestMemory(pinnedBytes, pinnedPtr,
                                                gpuBytes, gpuPtr);
}

/**
 *
 */
void      CudaBackend::releaseGpuMemory(void** pinnedPtr, void** gpuPtr) {
    CudaMemoryManager::instance().releaseMemory(pinnedPtr, gpuPtr);
}

/**
 *
 */
void      CudaBackend::reset(void) {
    CudaMemoryManager::instance().reset();
}

}

