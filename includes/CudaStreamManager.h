/**
 * \file    CudaStreamManager.h
 *
 * \brief  Write this
 *
 * Write this.  The class must be thread-safe as data distributor objects will
 * call a stream manager to request a stream and threads responsible for freeing
 * CudaDataPacket resources will call the manager to release streams.
 *
 */

#ifndef CUDA_STREAM_MANAGER_H__
#define CUDA_STREAM_MANAGER_H__

#include <deque>
#include <vector>
#include <pthread.h>

#include <driver_types.h>

#include "CudaStream.h"

namespace orchestration {

class CudaStreamManager {
public:
    ~CudaStreamManager();

    static void                 setMaxNumberStreams(const int nMaxStreams);
    static CudaStreamManager&   instance(void);

    int             numberFreeStreams(void);

    CudaStream      requestStream(const bool block);
    void            releaseStream(CudaStream& stream);

private:
    CudaStreamManager();

    static int                  nMaxStreams_;
    static bool                 wasInstantiated_;

    std::vector<cudaStream_t>   streams_;

    // TODO: Determine what the best data structure is.  Since this queue isn't
    // used directly by the threads in a ThreadTeam nor a thread that could
    // temporarily block the threads in a ThreadTeam, it may not be so
    // important.
    std::deque<CudaStream>      freeStreams_;

    pthread_mutex_t   idxMutex_;         //!< Use to access queue of free streams
    pthread_cond_t    streamReleased_;   //!< To be emitted when a thread releases a stream
};

}

#endif

