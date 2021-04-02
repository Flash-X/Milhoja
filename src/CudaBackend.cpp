#ifndef USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

#include "CudaBackend.h"

#include "CudaGpuEnvironment.h"
#include "StreamManager.h"
//#include "CudaStreamManager.h"
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
    orchestration::CudaGpuEnvironment::instantiate();
    //  Eventually this should just instantiate directly the CUDA SM.
    orchestration::StreamManager::instantiate(nStreams);
    orchestration::CudaMemoryManager::instantiate(nBytesInMemoryPools);
}

}

