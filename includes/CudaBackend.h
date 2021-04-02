#ifndef CUDA_BACKEND_H__
#define CUDA_BACKEND_H__

#include "Backend.h"

namespace orchestration {

class CudaBackend : public Backend {
public:
    ~CudaBackend(void)       {};

    CudaBackend(CudaBackend&)                  = delete;
    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend(CudaBackend&&)                 = delete;
    CudaBackend& operator=(CudaBackend&)       = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;
    CudaBackend& operator=(CudaBackend&&)      = delete;

    inline Stream    requestStream(const bool block) override;
    inline void      releaseStream(Stream& stream) override;
    inline void      requestMemory(const std::size_t bytes,
                                   void** hostPtr, void** gpuPtr) override;
    inline void      releaseMemory(void** hostPtr, void** gpuPtr) override;
    inline void      reset(void) override;

private:
    CudaBackend(const unsigned int nStreams,
                const std::size_t nBytesInMemoryPools);

    // Needed for polymorphic singleton
    friend Backend&   Backend::instance();
};

}

#endif

