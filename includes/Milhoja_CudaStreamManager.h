/**
 * \file    Milhoja_CudaStreamManager.h
 *
 * \brief  Write this
 *
 * Write this.  The class must be thread-safe as data distributor objects will
 * call a stream manager to request a stream and threads responsible for freeing
 * DataPacket resources will call the manager to release streams.
 *
 * The streams are a resource and as such all stream objects that are acquired
 * from a StreamManager must be released to the manager and without having been
 * modified (See documentation for Stream).  In addition, calling code must
 * not release Stream objects that were not acquired from the manager.
 *
 * \todo Determine what the best data structure is for streams_.  Since this
 * queue isn't used directly by the threads in a ThreadTeam nor a thread that
 * could temporarily block the threads in a ThreadTeam, it may not be so
 * important.
 *
 */

#ifndef MILHOJA_CUDA_STREAM_MANAGER_H__
#define MILHOJA_CUDA_STREAM_MANAGER_H__

#include <deque>

#include <pthread.h>

#include "Milhoja.h"
#include "Milhoja_Stream.h"

#ifndef MILHOJA_USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

namespace milhoja {

class CudaStreamManager {
public:
    ~CudaStreamManager();

    static void                 initialize(const int nMaxStreams);
    static CudaStreamManager&   instance(void);
    void                        finalize(void);

    int      maxNumberStreams(void) const  { return nMaxStreams_; };
    int      numberFreeStreams(void);

    Stream   requestStream(const bool block);
    void     releaseStream(Stream& stream);

private:
    CudaStreamManager();

    static bool   initialized_;
    static bool   finalized_;
    static int    nMaxStreams_;

    std::deque<Stream>   streams_;

    pthread_mutex_t   idxMutex_;         //!< Use to access queue of free streams
    pthread_cond_t    streamReleased_;   //!< To be emitted when a thread releases a stream
};

}

#endif

