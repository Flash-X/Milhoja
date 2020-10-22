// WIP: Somehow NDEBUG is getting set and deactivating the asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "CudaStreamManager.h"

#include <iostream>
#include <stdexcept>

#ifdef ENABLE_OPENACC_OFFLOAD
#include <openacc.h>
#endif

#include "OrchestrationLogger.h"

namespace orchestration {

// Default value chosen in conjunction with the error checking in the
// constructor such that client code will get an error if they do not explicitly
// set the numer of streams before accessing the manager.
int    CudaStreamManager::nMaxStreams_     = -1;
bool   CudaStreamManager::wasInstantiated_ = false;

/**
 * Before calling this routine, client code must first set the number of streams
 * to be managed using setMaxNumberStreams().
 *
 * \return 
 */
CudaStreamManager&   CudaStreamManager::instance(void) {
    if (nMaxStreams_ <= 0) {
        throw std::invalid_argument("[CudaStreamManager::instance] "
                                    "Set max number of streams before accessing manager");
    }

    static CudaStreamManager   stream_manager;
    return stream_manager;
}

/**
 * This member must be called before accessing the manager, but cannot be called
 * after accessing the manager.
 *
 * \return 
 */
void CudaStreamManager::setMaxNumberStreams(const int nMaxStreams) {
    if (wasInstantiated_) {
        throw std::logic_error("[CudaStreamManager::setMaxNumberStreams] "
                               "Cannot be set once the manager has been accessed");
    } else if (nMaxStreams <= 0) {
        // We need at least one stream to avoid deadlocking in requestStream
        // when there are no free streams.
        throw std::invalid_argument("[CudaStreamManager::setMaxNumberStreams] "
                                    "Need at least one stream");
    }

    nMaxStreams_ = nMaxStreams;
    Logger::instance().log( "[CudaStreamManager] Number of streams set to "
                           + std::to_string(nMaxStreams_));
}

/**
 * 
 *
 * \return 
 */
CudaStreamManager::CudaStreamManager(void)
    : streams_{nMaxStreams_},
      freeStreams_{}
{
    Logger::instance().log("[CudaStreamManager] Initializing...");
    assert(freeStreams_.size() == 0);

    pthread_cond_init(&streamReleased_, NULL);
    pthread_mutex_init(&idxMutex_, NULL);

    pthread_mutex_lock(&idxMutex_);

#ifdef ENABLE_OPENACC_OFFLOAD
    for (int i=0; i<streams_.size(); ++i) {
         int   streamId = i + 1;
         streams_[i] = static_cast<cudaStream_t>(acc_get_cuda_stream(streamId));

         freeStreams_.push_back( CudaStream(streamId, &(streams_[i])) );
    }
    Logger::instance().log(  "[CudaStreamManager] Acquired " 
                           + std::to_string(streams_.size())
                           + " CUDA streams from OpenACC asynchronous queues");
#else
    cudaError_t   cErr = cudaErrorInvalidValue;
    for (std::size_t i=0; i<streams_.size(); ++i) {
         cErr = cudaStreamCreate(&(streams_[i]));
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaStreamManager::CudaStreamManager] ";
            errMsg += "Unable to create CUDA stream\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            pthread_mutex_unlock(&idxMutex_);
            throw std::runtime_error(errMsg);
         }

         // Make stream indices 1-based so that 0 can work as NULL_STREAM
         freeStreams_.push_back( CudaStream(i+1, &(streams_[i])) );
    }
    Logger::instance().log(  "[CudaStreamManager] Created "
                           + std::to_string(streams_.size())
                           + " CUDA streams");
#endif

    wasInstantiated_ = true;
    Logger::instance().log("[CudaStreamManager] Created and ready for use");

    pthread_mutex_unlock(&idxMutex_);
}

/**
 * 
 *
 * \return 
 */
CudaStreamManager::~CudaStreamManager(void) {
    Logger::instance().log("[CudaStreamManager] Finalizing...");

    pthread_mutex_lock(&idxMutex_);

    // TODO: When designing an appropriate error handling system, should we
    // include the possibility of including warnings?  Should there be a logging
    // system to which we could write this?  Should the logging system patch
    // into a logging system offered by client code?
    if (freeStreams_.size() != streams_.size()) {
        std::cerr << "[CudaStreamManager::~CudaStreamManager] WARNING - "
                  << (streams_.size() - freeStreams_.size()) 
                  << " out of " << streams_.size()
                  << " streams have not been released" << std::endl;
    }

#ifdef ENABLE_OPENACC_OFFLOAD
    Logger::instance().log(  "[CudaStreamManager] No longer using "
                           + std::to_string(streams_.size())
                           + " CUDA streams/OpenACC asynchronous queues");
#else
    cudaError_t   cErr = cudaErrorInvalidValue;
    for (std::size_t i=0; i<streams_.size(); ++i) {
         cErr = cudaStreamDestroy(streams_[i]);
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaStreamManager::~CudaStreamManager] ";
            errMsg += "Unable to destroy CUDA stream\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            std::cerr << errMsg;
         }
    }
    Logger::instance().log(  "[CudaStreamManager] Destroyed "
                           + std::to_string(streams_.size())
                           + " CUDA streams");
#endif

    pthread_mutex_unlock(&idxMutex_);

    pthread_cond_destroy(&streamReleased_);
    pthread_mutex_destroy(&idxMutex_);

    wasInstantiated_ = false;
    Logger::instance().log("[CudaStreamManager] Destroyed");
}

/**
 * 
 *
 * \return 
 */
int  CudaStreamManager::numberFreeStreams(void) {
    pthread_mutex_lock(&idxMutex_);
    std::size_t nStreams = freeStreams_.size();
    pthread_mutex_unlock(&idxMutex_);

    if (nStreams > INT_MAX) {
        std::string  errMsg = "[CudaStreamManager::numberFreeStreams] ";
        errMsg += "Too many streams created\n";
        throw std::overflow_error(errMsg);
    }

    return static_cast<int>(nStreams);
}

/**
 * This should be by move only!
 *
 * If block is true and there are no free streams, this member function will
 * block the calling thread until a stream becomes free.
 *
 * \return The free stream that has been given to the calling code for exclusive
 * use.  If block is set to false and there are no free streams, then a null
 * stream object is returned.
 */
CudaStream    CudaStreamManager::requestStream(const bool block) {
    // Get exclusive access to the free stream queue so that we can safely get
    // the ID of a free stream from it.  It is also important for the case when
    // we need to wait for a streamReleased signal.  In particular, we need to
    // know that no thread can emit that signal between when we acquire the
    // mutex and when we begin to wait for the signal.  In other words, we won't
    // miss the signal, which could be important if there is presently only one
    // stream in use by client code.
    pthread_mutex_lock(&idxMutex_);

    if (freeStreams_.size() <= 0) {
        if (block) {
            // Block until a stream is released and this thread hits the 
            // lottery and actually gets to take control of it.
            //
            // To avoid deadlocking on this wait, we need at least one stream
            // out for business.  Given that we know that the number of free
            // streams is zero, there can be no deadlock so long as this object
            // is managing at least one stream.
            do {
                Logger::instance().log("[CudaStreamManager] No streams available.  Blocking as requested.");
                pthread_cond_wait(&streamReleased_, &idxMutex_);
                // TODO: Add this back in but only when desired verbosity is
                // high.
//                Logger::instance().log("[CudaStreamManager] Stream has been released");
            } while(freeStreams_.size() <= 0);
        } else {
            Logger::instance().log("[CudaStreamManager] No streams available. Returning null stream as requested.");
            pthread_mutex_unlock(&idxMutex_);

            return CudaStream{};
        }
    }

    // The stream IDs are 1-based, but the queue is 0-based
    CudaStream   stream = std::move(freeStreams_.front());
    freeStreams_.pop_front();

    pthread_mutex_unlock(&idxMutex_);

    if ( stream.object != &(streams_[stream.id-1]) ) {
        throw std::invalid_argument("[CudaStreamManager::requestStream] "
                                    "Given stream ID and pointer not properly matched");
    }

    return stream;
}

/**
 * 
 *
 * \return 
 */
void   CudaStreamManager::releaseStream(CudaStream& stream) {
    if        (stream.id == CudaStream::NULL_STREAM_ID) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Given stream has null ID");
    } else if (stream.id > streams_.size()) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Stream ID is too large");
    } else if (stream.object == nullptr) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Given stream has null object");
    } else if (stream.object != &(streams_[stream.id-1]) ) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Given stream ID and pointer not properly matched");
    }

    pthread_mutex_lock(&idxMutex_);

    if (freeStreams_.size() >= streams_.size()) {
        pthread_mutex_unlock(&idxMutex_);
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "All streams accounted for.  No streams to release.");
    }

    for (const auto& freeStream : freeStreams_) {
        if (stream.id == freeStream.id) {
            pthread_mutex_unlock(&idxMutex_);
            throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                        "Given stream is already free");
        }
    }

    // We must put the stream back in the queue before emitting the signal
    freeStreams_.push_back( std::move(stream) );
    pthread_cond_signal(&streamReleased_);

    pthread_mutex_unlock(&idxMutex_);
}

}

