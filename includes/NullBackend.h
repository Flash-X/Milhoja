#ifndef NULL_BACKEND_H__
#define NULL_BACKEND_H__

#include <stdexcept>

#include "Backend.h"

namespace orchestration {

class NullBackend : public Backend {
public:
    ~NullBackend(void)       {};

    NullBackend(NullBackend&)                  = delete;
    NullBackend(const NullBackend&)            = delete;
    NullBackend(NullBackend&&)                 = delete;
    NullBackend& operator=(NullBackend&)       = delete;
    NullBackend& operator=(const NullBackend&) = delete;
    NullBackend& operator=(NullBackend&&)      = delete;

    int       maxNumberStreams(void) const override
        { throw std::logic_error("[NullBackend::maxNumberStreams] Not implemented"); };
    int       numberFreeStreams(void) override
        { throw std::logic_error("[NullBackend::numberFreeStreams] Not implemented"); };
    Stream    requestStream(const bool block) override
        { throw std::logic_error("[NullBackend::requestStream] Not implemented"); };
    void      releaseStream(Stream& stream) override
        { throw std::logic_error("[NullBackend::releaseStream] Not implemented"); };

    void      initiateHostToGpuTransfer(DataPacket& packet) override
        { throw std::logic_error("[NullBackend::initiateHostToGpuTransfer] Not implemented"); };
    void      initiateGpuToHostTransfer(DataPacket& packet,
                                        cudaHostFn_t callback,
                                        void* callbackData) override
        { throw std::logic_error("[NullBackend::initiateGpuToHostTransfer] Not implemented"); };

    void      requestMemory(const std::size_t bytes,
                            void** hostPtr, void** gpuPtr) override
        { throw std::logic_error("[NullBackend::requestMemory] Not implemented"); };
    void      releaseMemory(void** hostPtr, void** gpuPtr) override
        { throw std::logic_error("[NullBackend::releaseMemory] Not implemented"); };
    void      reset(void) override
        { throw std::logic_error("[NullBackend::reset] Not implemented"); };

private:
    NullBackend(void)        {};

    // Needed for polymorphic singleton
    friend Backend&   Backend::instance();
};

}

#endif

