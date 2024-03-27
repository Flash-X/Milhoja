#include "Milhoja_FakeCudaStreamManager.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <climits>

#include "Milhoja.h"
#include "Milhoja_Logger.h"

#ifdef MILHOJA_OPENACC_OFFLOADING
#include <openacc.h>
#endif

namespace milhoja {

bool   FakeCudaStreamManager::initialized_ = false;
bool   FakeCudaStreamManager::finalized_   = false;
int    FakeCudaStreamManager::nMaxStreams_ = -1;

/**
 * Instantiate and initialize the application's singleton stream manager object.
 *
 * \param nMaxStreams - the maximum number of streams to be made available.  The
 *                      given value must be a positive integer.
 */
void   FakeCudaStreamManager::initialize(const int nMaxStreams) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[FakeCudaStreamManager::initialize] "
                               "Already initialized");
    } else if (nMaxStreams <= 0) {
        // We need at least one stream to avoid deadlocking in requestStream
        // when there are no free streams.
        throw std::invalid_argument("[FakeCudaStreamManager::initialize] "
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
void    FakeCudaStreamManager::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[FakeCudaStreamManager::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[FakeCudaStreamManager::finalize] Already finalized");
    }

    Logger::instance().log("[FakeCudaStreamManager] Finalizing ...");

    pthread_mutex_lock(&idxMutex_);

    if (streams_.size() != nMaxStreams_) {
        std::string   errMsg =   "[FakeCudaStreamManager::finalize] "
                               + std::to_string(nMaxStreams_ - streams_.size()) 
                               + " out of "
                               + std::to_string(nMaxStreams_)
                               + " streams have not been released";
        throw std::runtime_error(errMsg);
    }

#ifdef MILHOJA_OPENACC_OFFLOADING
    Logger::instance().log(  "[FakeCudaStreamManager] No longer using "
                           + std::to_string(streams_.size())
                           + " FAKECUDA streams/OpenACC asynchronous queues");
#else
    cudaError_t   cErr = cudaErrorInvalidValue;
    for (std::size_t i=0; i<streams_.size(); ++i) {
         cErr = cudaStreamDestroy(streams_[i].cudaStream);
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[FakeCudaStreamManager::finalize] ";
            errMsg += "Unable to destroy CUDA stream\n";
            errMsg += "FAKECUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr));
            throw std::runtime_error(errMsg);
         }
    }
    Logger::instance().log(  "[FakeCudaStreamManager] Destroyed "
                           + std::to_string(streams_.size())
                           + " FAKECUDA streams");
#endif

    pthread_mutex_unlock(&idxMutex_);

    pthread_cond_destroy(&streamReleased_);
    pthread_mutex_destroy(&idxMutex_);

    nMaxStreams_ = -1;

    finalized_ = true;

    Logger::instance().log("[FakeCudaStreamManager] Finalized");
}

/**
 * Request access to the singleton stream manager.  Before calling this routine,
 * calling code must first initialize the manager.
 */
FakeCudaStreamManager&   FakeCudaStreamManager::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[FakeCudaStreamManager::instance] Singleton not initialized");
    } else if (finalized_) {
        throw std::logic_error("[FakeCudaStreamManager::instance] No access after finalization");
    }

    static FakeCudaStreamManager   manager;
    return manager;
}

/**
 * 
 *
 * \return 
 */
FakeCudaStreamManager::FakeCudaStreamManager(void)
    : streams_{}
{
    Logger::instance().log("[FakeCudaStreamManager] Initializing...");
    assert(streams_.size() == 0);

    pthread_cond_init(&streamReleased_, NULL);
    pthread_mutex_init(&idxMutex_, NULL);

    pthread_mutex_lock(&idxMutex_);

#ifdef MILHOJA_OPENACC_OFFLOADING
    Stream         stream{};
    for (int i=0; i<nMaxStreams_; ++i) {
         stream.accAsyncQueue = i + 1;
         // stream.cudaStream = static_cast<cudaStream_t>(acc_get_cuda_stream(stream.accAsyncQueue));
         // if (stream.cudaStream == nullptr) {
         //    std::string  errMsg = "[FakeCudaStreamManager::FakeCudaStreamManager] ";
         //    errMsg += "CUDA stream is null\n";
         //    throw std::runtime_error(errMsg);
         // }

         streams_.push_back( std::move(stream) );
    }
    Logger::instance().log(  "[FakeCudaStreamManager] Acquired " 
                           + std::to_string(streams_.size())
                           + " CUDA streams from OpenACC asynchronous queues");
#else
    Stream   stream{};
    for (std::size_t i=0; i<nMaxStreams_; ++i) {
         cudaError_t    cErr = cudaStreamCreate(&(stream.cudaStream));
         if (cErr != cudaSuccess) {
            std::string  errMsg = "[FakeCudaStreamManager::FakeCudaStreamManager] ";
            errMsg += "Unable to create CUDA stream\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            pthread_mutex_unlock(&idxMutex_);
            throw std::runtime_error(errMsg);
         }

         streams_.push_back( std::move(stream) );
    }
    Logger::instance().log(  "[FakeCudaStreamManager] Created "
                           + std::to_string(streams_.size())
                           + " FAKECUDA streams");
#endif

    Logger::instance().log("[FakeCudaStreamManager] Created and ready for use");

    pthread_mutex_unlock(&idxMutex_);
}

/**
 * Based on the StreamManager's singleton design pattern, this will only be
 * called at program termination.
 */
FakeCudaStreamManager::~FakeCudaStreamManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[FakeCudaStreamManager::~FakeCudaStreamManager] ERROR - Not finalized"
                  << std::endl;
    }
}

/**
 * 
 *
 * \return 
 */
int  FakeCudaStreamManager::numberFreeStreams(void) {
    pthread_mutex_lock(&idxMutex_);
    std::size_t nStreams = streams_.size();
    pthread_mutex_unlock(&idxMutex_);

    if (nStreams > INT_MAX) {
        std::string  errMsg = "[FakeCudaStreamManager::numberFreeStreams] ";
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
 * \todo   If, as is the case for FAKECUDA, the streams are relatively cheap
 *         objects, then should we allow this routine to allocate more streams
 *         rather than block?  This could be helpful to prevent possible
 *         deadlocks when a code needs to acquire more than one stream.
 *         However, we shouldn't make this a feature of the general design
 *         concept unless we know that all stream managers can dynamically grow
 *         their reserve.
 * \todo   Add in logging of release if verbosity level is high enough.
 */
Stream    FakeCudaStreamManager::requestStream(const bool block) {
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
                Logger::instance().log("[FakeCudaStreamManager] No streams available.  Blocking as requested.");
                pthread_cond_wait(&streamReleased_, &idxMutex_);
//                Logger::instance().log("[FakeCudaStreamManager] Stream has been released");
            } while(streams_.size() <= 0);
        } else {
            Logger::instance().log("[FakeCudaStreamManager] No streams available. Returning null stream as requested.");
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
void   FakeCudaStreamManager::releaseStream(Stream& stream) {
    // if (stream.cudaStream == nullptr) {
    //     throw std::invalid_argument("[FakeCudaStreamManager::releaseStream] "
    //                                 "Given stream has null CUDA stream");
    // }
#ifdef MILHOJA_OPENACC_OFFLOADING
    if (stream.accAsyncQueue == NULL_ACC_ASYNC_QUEUE) {
        throw std::invalid_argument("[FakeCudaStreamManager::releaseStream] "
                                    "Given stream has null OpenACC asynchronous queue");
#endif
    }

    pthread_mutex_lock(&idxMutex_);

    if (streams_.size() >= nMaxStreams_) {
        pthread_mutex_unlock(&idxMutex_);
        throw std::invalid_argument("[FakeCudaStreamManager::releaseStream] "
                                    "All streams accounted for.  No streams to release.");
    }

#ifdef DEBUG_RUNTIME
    // Streams will be released frequently and we might have a great many
    // streams.  Therefore, we don't want to perform this error checking by
    // default.
    for (const auto& freeStream : freeStreams_) {
        if (stream.cudaStream == freeStream.cudaStream) {
            pthread_mutex_unlock(&idxMutex_);
            throw std::invalid_argument("[FakeCudaStreamManager::releaseStream] "
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

