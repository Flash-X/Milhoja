#ifndef MILHOJA_FAKECUDA_BACKEND_H__
#define MILHOJA_FAKECUDA_BACKEND_H__

#include "Milhoja.h"
#include "Milhoja_RuntimeBackend.h"

#ifndef MILHOJA_HOSTMEM_RUNTIME_BACKEND
#error "This file need not be compiled if the HOSTMEM backend isn't used"
#endif

namespace milhoja {

/**
 * \class FakeCudaBackend Milhoja_FakeCudaBackend.h
 *
 * Provide a FAKECUDA-based concrete implementation of the RuntimeBackend class.
 * Please refer to the documentation for RuntimeBackend class for more
 * information.
 */
class FakeCudaBackend : public RuntimeBackend {
public:
    ~FakeCudaBackend(void)       {};

    FakeCudaBackend(FakeCudaBackend&)                  = delete;
    FakeCudaBackend(const FakeCudaBackend&)            = delete;
    FakeCudaBackend(FakeCudaBackend&&)                 = delete;
    FakeCudaBackend& operator=(FakeCudaBackend&)       = delete;
    FakeCudaBackend& operator=(const FakeCudaBackend&) = delete;
    FakeCudaBackend& operator=(FakeCudaBackend&&)      = delete;

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
    FakeCudaBackend(const unsigned int nStreams,
                const std::size_t nBytesInMemoryPools);

    // Needed for polymorphic singleton
    friend RuntimeBackend&   RuntimeBackend::instance();
};

}

#endif

