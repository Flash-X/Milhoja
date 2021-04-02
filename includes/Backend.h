#ifndef BACKEND_H__
#define BACKEND_H__

#include <cstddef>

#include "Stream.h"

namespace orchestration {

class Backend {
public:
    virtual ~Backend(void)     { instantiated_ = false; };

    Backend(Backend&)                  = delete;
    Backend(const Backend&)            = delete;
    Backend(Backend&&)                 = delete;
    Backend& operator=(Backend&)       = delete;
    Backend& operator=(const Backend&) = delete;
    Backend& operator=(Backend&&)      = delete;

    static void     instantiate(const unsigned int nStreams,
                                const std::size_t  nBytesInMemoryPools);
    static Backend& instance(void);
 
    virtual int       maxNumberStreams(void) const = 0; 
    virtual int       numberFreeStreams(void) = 0;
    virtual Stream    requestStream(const bool block) = 0;
    virtual void      releaseStream(Stream& stream) = 0;
    virtual void      requestMemory(const std::size_t bytes,
                                    void** hostPtr, void** gpuPtr) = 0;
    virtual void      releaseMemory(void** hostPtr, void** gpuPtr) = 0;
    virtual void      reset(void) = 0;
    // FIXME: This is temprorary since this manager is so rudimentary

    // TODO: See about taking data movements out of packet.
    //       Given a packet, the backend could do transfers.
    //       This could simplify the design/implementation of
    //       packets in the runtime library.

protected:
    Backend(void)     {};

private:
    static bool           instantiated_;
    static unsigned int   nStreams_;
    static std::size_t    nBytesInMemoryPools_;
};

}

#endif

