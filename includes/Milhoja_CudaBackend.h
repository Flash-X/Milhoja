#ifndef MILHOJA_CUDA_BACKEND_H__
#define MILHOJA_CUDA_BACKEND_H__

#include "Milhoja.h"
#include "Milhoja_RuntimeBackend.h"

#ifndef MILHOJA_USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

namespace milhoja {

/**
 * \class CudaBackend Milhoja_CudaBackend.h
 *
 * Provide a CUDA-based concrete implementation of the RuntimeBackend class.
 * Please refer to the documentation for RuntimeBackend class for more
 * information.
 */
class CudaBackend : public RuntimeBackend {
public:
    ~CudaBackend(void)       {};

    CudaBackend(CudaBackend&)                  = delete;
    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend(CudaBackend&&)                 = delete;
    CudaBackend& operator=(CudaBackend&)       = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;
    CudaBackend& operator=(CudaBackend&&)      = delete;

    void      finalize(void) override;

    int       maxNumberStreams(void) const override; 
    int       numberFreeStreams(void) override;
    Stream    requestStream(const bool block) override;
    void      releaseStream(Stream& stream) override;

    void      initiateHostToGpuTransfer(DataPacket& packet) override;
    void      initiateGpuToHostTransfer(DataPacket& packet,
                                        GPU_TO_HOST_CALLBACK_FCN callback,
                                        void* callbackData) override;

    void      requestGpuMemory(const std::size_t pinnedBytes,
                               void** pinnedPtr,
                               const std::size_t gpuBytes,
                               void** gpuPtr);
    void      releaseGpuMemory(void** pinnedPtr, void** gpuPtr) override;
    void      reset(void) override;

private:
    CudaBackend(const unsigned int nStreams,
                const std::size_t nBytesInMemoryPools);

    // Needed for polymorphic singleton
    friend RuntimeBackend&   RuntimeBackend::instance();
};

}

#endif

