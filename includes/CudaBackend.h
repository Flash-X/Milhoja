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

    int       maxNumberStreams(void) const override; 
    int       numberFreeStreams(void) override;
    Stream    requestStream(const bool block) override;
    void      releaseStream(Stream& stream) override;
    void      requestMemory(const std::size_t bytes,
                                   void** hostPtr, void** gpuPtr) override;
    void      releaseMemory(void** hostPtr, void** gpuPtr) override;
    void      reset(void) override;

private:
    CudaBackend(const unsigned int nStreams,
                const std::size_t nBytesInMemoryPools);

    // Needed for polymorphic singleton
    friend Backend&   Backend::instance();
};

}

#endif

