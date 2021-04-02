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
void      CudaBackend::requestMemory(const std::size_t bytes,
                                     void** hostPtr, void** gpuPtr) {
    CudaMemoryManager::instance().requestMemory(bytes, hostPtr, gpuPtr);
}

/**
 *
 */
void      CudaBackend::releaseMemory(void** hostPtr, void** gpuPtr) {
    CudaMemoryManager::instance().releaseMemory(hostPtr, gpuPtr);
}

/**
 *
 */
void      CudaBackend::reset(void) {
    CudaMemoryManager::instance().reset();
}

}

