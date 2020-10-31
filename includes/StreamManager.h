/**
 * \file    StreamManager.h
 *
 * \brief  Write this
 *
 * The streams are a resource and as such all stream objects that are acquired
 * from a StreamManager must be released to the manager and without having been
 * modified (See documentation for Stream).  In addition, calling code must
 * not release Stream objects that were not acquired from the manager.
 *
 */

#ifndef STREAM_MANAGER_H__
#define STREAM_MANAGER_H__

#include "Stream.h"

namespace orchestration {

class StreamManager {
public:
    virtual ~StreamManager()    { instantiated_ = false; };

    static void             instantiate(const int nMaxStreams);
    static StreamManager&   instance(void);

    int             maxNumberStreams(void) const { return nMaxStreams_; }
    virtual int     numberFreeStreams(void) = 0;

    virtual Stream  requestStream(const bool block) = 0;
    virtual void    releaseStream(Stream& stream) = 0;

protected:
    StreamManager()   {  };

    static int      nMaxStreams_;

private:
    static bool     instantiated_;
};

}

#endif

