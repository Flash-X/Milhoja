#ifndef MILHOJA_HIP_BACKEND_H__
#define MILHOJA_HIP_BACKEND_H__

#include "Milhoja.h"
#include "Milhoja_RuntimeBackend.h"

#ifndef MILHOJA_HIP_RUNTIME_BACKEND
#error "This file need not be compiled if the HIP backend is not used"
#endif

namespace milhoja {

class HipBackend : public RuntimeBackend {
public:
    ~HipBackend(void) = default;

    HipBackend(HipBackend&) = delete;
    HipBackend(const HipBackend&) = delete;
    HipBackend(HipBackend&&) = delete;
    HipBackend& operator=(HipBackend&) = delete;
    HipBackend& operator=(const HipBackend&) = delete;
    HipBackend& operator=(HipBackend&&) = delete;

    void finalize(void) override;

    int maxNumberStreams(void) const override;
    int numberFreeStreams(void) override;
    Stream requestStream(const bool block) override;
    void releaseStream(Stream& stream) override;
    void synchronizeStream(const Stream& stream) override;

    void initiateHostToGpuTransfer(DataPacket& packet) override;
    void initiateGpuToHostTransfer(DataPacket& packet,
                                   GPU_TO_HOST_CALLBACK_FCN callback,
                                   void* callbackData) override;

    void requestGpuMemory(const std::size_t pinnedBytes,
                          void** pinnedPtr,
                          const std::size_t gpuBytes,
                          void** gpuPtr) override;
    void releaseGpuMemory(void** pinnedPtr, void** gpuPtr) override;
    void reset(void) override;

private:
    HipBackend(const unsigned int nStreams,
               const std::size_t nBytesInMemoryPools);

    friend RuntimeBackend& RuntimeBackend::instance();
};

}

#endif
