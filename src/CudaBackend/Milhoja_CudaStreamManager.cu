#include "Milhoja_CudaStreamManager.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Milhoja.h"
#include "Milhoja_Logger.h"

#ifdef MILHOJA_OPENACC_OFFLOADING
#include <openacc.h>
#endif

namespace milhoja {

bool   CudaStreamManager::initialized_ = false;
bool   CudaStreamManager::finalized_   = false;
int    CudaStreamManager::nMaxStreams_ = -1;

/**
 * Instantiate and initialize the application's singleton stream manager object.
 *
 * \param nMaxStreams - the maximum number of streams to be made available.  The
 *                      given value must be a positive integer.
 */
void   CudaStreamManager::initialize(const int nMaxStreams) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[CudaStreamManager::initialize] "
                               "Already initialized");
    } else if (nMaxStreams <= 0) {
        // We need at least one stream to avoid deadlocking in requestStream
        // when there are no free streams.
        throw std::invalid_argument("[CudaStreamManager::initialize] "
                                    "Need at least one stream");
    }

    // Create/initialize
    nMaxStreams_ = nMaxStreams;
    initialized_ = true;

    instance();
}

/**
 * 
 */
void    CudaStreamManager::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[CudaStreamManager::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[CudaStreamManager::finalize] Already finalized");
    }

    Logger::instance().log("[CudaStreamManager] Finalizing ...");

    pthread_mutex_lock(&idxMutex_);

    if (streams_.size() != nMaxStreams_) {
        std::string   errMsg =   "[CudaStreamManager::finalize] "
                               + std::to_string(nMaxStreams_ - streams_.size()) 
                               + " out of "
                               + std::to_string(nMaxStreams_)
                               + " streams have not been released";
        throw std::runtime_error(errMsg);
    }

#ifdef MILHOJA_OPENACC_OFFLOADING
    Logger::instance().log(  "[CudaStreamManager] No longer using "
                           + std::to_string(streams_.size())
                           + " CUDA streams/OpenACC asynchronous queues");
#else
    cudaError_t   cErr = cudaErrorInvalidValue;
    for (std::size_t i=0; i<streams_.size(); ++i) {
         cErr = cudaStreamDestroy(streams_[i].cudaStream);
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaStreamManager::finalize] ";
            errMsg += "Unable to destroy CUDA stream\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr));
            throw std::runtime_error(errMsg);
         }
    }
    Logger::instance().log(  "[CudaStreamManager] Destroyed "
                           + std::to_string(streams_.size())
                           + " CUDA streams");
#endif

    pthread_mutex_unlock(&idxMutex_);

    pthread_cond_destroy(&streamReleased_);
    pthread_mutex_destroy(&idxMutex_);

    nMaxStreams_ = -1;

    finalized_ = true;

    Logger::instance().log("[CudaStreamManager] Finalized");
}

/**
 * Request access to the singleton stream manager.  Before calling this routine,
 * calling code must first initialize the manager.
 */
CudaStreamManager&   CudaStreamManager::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[CudaStreamManager::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[CudaStreamManager::instance] No access after finalization");
    }

    static CudaStreamManager   manager;
    return manager;
}

/**
 * 
 *
 * \return 
 */
CudaStreamManager::CudaStreamManager(void)
    : streams_{}
{
    Logger::instance().log("[CudaStreamManager] Initializing...");
    assert(streams_.size() == 0);

    pthread_cond_init(&streamReleased_, NULL);
    pthread_mutex_init(&idxMutex_, NULL);

    pthread_mutex_lock(&idxMutex_);

#ifdef MILHOJA_OPENACC_OFFLOADING
    Stream         stream{};
    for (int i=0; i<nMaxStreams_; ++i) {
         stream.accAsyncQueue = i + 1;
         stream.cudaStream = static_cast<cudaStream_t>(acc_get_cuda_stream(stream.accAsyncQueue));
         if (stream.cudaStream == nullptr) {
            std::string  errMsg = "[CudaStreamManager::CudaStreamManager] ";
            errMsg += "CUDA stream is null\n";
            throw std::runtime_error(errMsg);
         }

         streams_.push_back( std::move(stream) );
    }
    Logger::instance().log(  "[CudaStreamManager] Acquired " 
                           + std::to_string(streams_.size())
                           + " CUDA streams from OpenACC asynchronous queues");
#else
    Stream   stream{};
    for (std::size_t i=0; i<nMaxStreams_; ++i) {
         cudaError_t    cErr = cudaStreamCreate(&(stream.cudaStream));
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaStreamManager::CudaStreamManager] ";
            errMsg += "Unable to create CUDA stream\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            pthread_mutex_unlock(&idxMutex_);
            throw std::runtime_error(errMsg);
         }

         streams_.push_back( std::move(stream) );
    }
    Logger::instance().log(  "[CudaStreamManager] Created "
                           + std::to_string(streams_.size())
                           + " CUDA streams");
#endif

    Logger::instance().log("[CudaStreamManager] Created and ready for use");

    pthread_mutex_unlock(&idxMutex_);
}

/**
 * Based on the StreamManager's singleton design pattern, this will only be
 * called at program termination.
 */
CudaStreamManager::~CudaStreamManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[CudaStreamManager::~CudaStreamManager] ERROR - Not finalized"
                  << std::endl;
    }
}

/**
 * 
 *
 * \return 
 */
int  CudaStreamManager::numberFreeStreams(void) {
    pthread_mutex_lock(&idxMutex_);
    std::size_t nStreams = streams_.size();
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
 * Refer to the documentation of the requestStream member function of the
 * RuntimeBackend class.
 *
 * \todo   If, as is the case for CUDA, the streams are relatively cheap
 *         objects, then should we allow this routine to allocate more streams
 *         rather than block?  This could be helpful to prevent possible
 *         deadlocks when a code needs to acquire more than one stream.
 *         However, we shouldn't make this a feature of the general design
 *         concept unless we know that all stream managers can dynamically grow
 *         their reserve.  This, presently, cannot be implemented as
 *         OpenACC+CUDA with PGI on Summit has an upper limit of 32 streams.
 * \todo   Add in logging of release if verbosity level is high enough.
 */
Stream    CudaStreamManager::requestStream(const bool block) {
    // Get exclusive access to the free stream queue so that we can safely get
    // the ID of a free stream from it.  It is also important for the case when
    // we need to wait for a streamReleased signal.  In particular, we need to
    // know that no thread can emit that signal between when we acquire the
    // mutex and when we begin to wait for the signal.  In other words, we won't
    // miss the signal, which could be important if there is presently only one
    // stream in use by client code.
    pthread_mutex_lock(&idxMutex_);

    if (streams_.size() <= 0) {
        if (block) {
            // Block until a stream is released and this thread hits the 
            // lottery and actually gets to take control of it.
            //
            // To avoid deadlocking on this wait, we need at least one stream
            // out for business.  Given that we know that the number of free
            // streams is zero, there can be no deadlock so long as this object
            // is managing at least one stream.
            //
            // There exists another possibility for deadlocking based on the
            // notion that any code can request a stream.  Consider the case of
            // 5 streams total and five data packets that request and receive
            // one stream each.  If each action associated with the data packets
            // subsequently request an extra stream, then they will all wait for
            // a free stream and none will therefore release a stream.
            do {
                Logger::instance().log("[CudaStreamManager] No streams available.  Blocking as requested.");
                pthread_cond_wait(&streamReleased_, &idxMutex_);
//                Logger::instance().log("[CudaStreamManager] Stream has been released");
            } while(streams_.size() <= 0);
        } else {
            Logger::instance().log("[CudaStreamManager] No streams available. Returning null stream as requested.");
            pthread_mutex_unlock(&idxMutex_);

            return Stream{};
        }
    }

    Stream   stream = std::move(streams_.front());
    streams_.pop_front();

    pthread_mutex_unlock(&idxMutex_);

    return stream;
}

/**
 * Refer to the documentation of the releaseStream member function of the
 * RuntimeBackend class.
 */
void   CudaStreamManager::releaseStream(Stream& stream) {
    if (stream.cudaStream == nullptr) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Given stream has null CUDA stream");
#ifdef MILHOJA_OPENACC_OFFLOADING
    } else if (stream.accAsyncQueue == NULL_ACC_ASYNC_QUEUE) {
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "Given stream has null OpenACC asynchronous queue");
#endif
    }

    pthread_mutex_lock(&idxMutex_);

    if (streams_.size() >= nMaxStreams_) {
        pthread_mutex_unlock(&idxMutex_);
        throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                    "All streams accounted for.  No streams to release.");
    }

#ifdef DEBUG_RUNTIME
    // Streams will be released frequently and we might have a great many
    // streams.  Therefore, we don't want to perform this error checking by
    // default.
    for (const auto& freeStream : streams_) {
        if (stream.cudaStream == freeStream.cudaStream) {
            pthread_mutex_unlock(&idxMutex_);
            throw std::invalid_argument("[CudaStreamManager::releaseStream] "
                                        "Given stream is already free");
        }
    }
#endif

    // We must put the stream back in the queue before emitting the signal
    streams_.push_back( std::move(stream) );
    pthread_cond_signal(&streamReleased_);

    pthread_mutex_unlock(&idxMutex_);
}

}

