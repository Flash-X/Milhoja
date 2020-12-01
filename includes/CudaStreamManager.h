/**
 * \file    CudaStreamManager.h
 *
 * \brief  Write this
 *
 * Write this.  The class must be thread-safe as data distributor objects will
 * call a stream manager to request a stream and threads responsible for freeing
 * CudaDataPacket resources will call the manager to release streams.
 *
 * \todo Determine what the best data structure is for streams_.  Since this
 * queue isn't used directly by the threads in a ThreadTeam nor a thread that
 * could temporarily block the threads in a ThreadTeam, it may not be so
 * important.
 *
 */

#ifndef CUDA_STREAM_MANAGER_H__
#define CUDA_STREAM_MANAGER_H__

#include <deque>
#include <pthread.h>

#include "StreamManager.h"

namespace orchestration {

class CudaStreamManager : public StreamManager {
public:
    ~CudaStreamManager();

    int      numberFreeStreams(void) override;

    Stream   requestStream(const bool block) override;
    void     releaseStream(Stream& stream) override;

private:
    CudaStreamManager();

    // Needed for polymorphic singleton
    friend StreamManager& StreamManager::instance();

    std::deque<Stream>   streams_;

    pthread_mutex_t   idxMutex_;         //!< Use to access queue of free streams
    pthread_cond_t    streamReleased_;   //!< To be emitted when a thread releases a stream
};

}

#endif

