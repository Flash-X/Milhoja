#ifndef MILHOJA_HIP_STREAM_MANAGER_H__
#define MILHOJA_HIP_STREAM_MANAGER_H__

#include <deque>
#include <pthread.h>

#include "Milhoja.h"
#include "Milhoja_Stream.h"

#ifndef MILHOJA_HIP_RUNTIME_BACKEND
#error "This file need not be compiled if the HIP backend is not used"
#endif

namespace milhoja {

class HipStreamManager {
public:
    ~HipStreamManager(void);

    static void initialize(const int nMaxStreams);
    static HipStreamManager& instance(void);
    void finalize(void);

    int maxNumberStreams(void) const { return nMaxStreams_; }
    int numberFreeStreams(void);
    Stream requestStream(const bool block);
    void releaseStream(Stream& stream);

private:
    HipStreamManager(void);

    static bool initialized_;
    static bool finalized_;
    static int nMaxStreams_;

    std::deque<Stream> streams_;
    pthread_mutex_t mutex_;
    pthread_cond_t streamReleased_;
};

}

#endif
