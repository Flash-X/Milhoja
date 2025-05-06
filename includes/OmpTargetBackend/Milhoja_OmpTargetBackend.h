#ifndef MILHOJA_OMPTARGET_BACKEND_H__
#define MILHOJA_OMPTARGET_BACKEND_H__

#include "Milhoja.h"
#include "Milhoja_RuntimeBackend.h"

#ifndef MILHOJA_OMPTARGET_RUNTIME_BACKEND
#error "This file need not be compiled if the OmpTarget backend isn't used"
#endif

namespace milhoja {

/**
 * \class OmpTargetBackend Milhoja_OmpTargetBackend.h
 *
 * Provide an OMP-taget-based concrete implementation of the RuntimeBackend class.
 * Please refer to the documentation for RuntimeBackend class for more
 * information.
 */
class OmpTargetBackend : public RuntimeBackend {
public:
    ~OmpTargetBackend(void)       {};

    OmpTargetBackend(OmpTargetBackend&)                  = delete;
    OmpTargetBackend(const OmpTargetBackend&)            = delete;
    OmpTargetBackend(OmpTargetBackend&&)                 = delete;
    OmpTargetBackend& operator=(OmpTargetBackend&)       = delete;
    OmpTargetBackend& operator=(const OmpTargetBackend&) = delete;
    OmpTargetBackend& operator=(OmpTargetBackend&&)      = delete;

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
    OmpTargetBackend(const unsigned int nStreams,
                const std::size_t nBytesInMemoryPools);

    // Needed for polymorphic singleton
    friend RuntimeBackend&   RuntimeBackend::instance();

    int                      target_device_num_;
};

}

#endif
