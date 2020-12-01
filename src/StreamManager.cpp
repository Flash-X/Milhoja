#include "StreamManager.h"

#include <stdexcept>

#ifdef USE_CUDA_BACKEND
#include "CudaStreamManager.h"
#endif

namespace orchestration {

int    StreamManager::nMaxStreams_ = -1;
bool   StreamManager::instantiated_ = false;

/**
 * Instantiate and initalize the application's singleton stream manager object.
 *
 * \param nMaxStreams - the maximum number of streams to be made available.  The
 *                      given value must be a positive integer.
 */
void   StreamManager::instantiate(const int nMaxStreams) {
    if (instantiated_) {
        throw std::logic_error("[StreamManager::instantiate] "
                               "StreamManager already instantiated");
    } else if (nMaxStreams <= 0) {
        // We need at least one stream to avoid deadlocking in requestStream
        // when there are no free streams.
        throw std::invalid_argument("[StreamManager::instantiate] "
                                    "Need at least one stream");
    }

    // Create/initialize
    nMaxStreams_ = nMaxStreams;
    instantiated_ = true;

    instance();
}

/**
 * Before calling this routine, client code must first instantiate the manager.
 *
 * \return A reference to the stream manager that is associated with the 
 *         runtime backend.
 */
StreamManager&   StreamManager::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[StreamManager::instance] "
                               "StreamManager must be instantiated first");
    }

#ifdef USE_CUDA_BACKEND
    static CudaStreamManager   manager;
#else
#error "Please specify a runtime backend"
#endif

    return manager;
}

}

