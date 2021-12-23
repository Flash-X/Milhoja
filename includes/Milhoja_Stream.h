/**
 * \file   Milhoja_Stream.h
 *
 * \brief  Write this
 *
 * This object should be designed and used such that there is no need to
 * express nor know what backend is used.  Similarly, it should not matter which
 * version of a stream is the base stream of the implementation in use.  In
 * other words, all the different types of streams present in a Stream object
 * are on equal footing.
 *
 * No code, except for StreamManagers, should alter the contents of a
 * pre-existing or given Stream object.  Unfortunately, the public contents
 * cannot be made const as CUDA routines that require a cudaStream_t do not
 * make the argument as const.
 *
 */

#ifndef MILHOJA_STREAM_H__
#define MILHOJA_STREAM_H__

#if defined(MILHOJA_USE_CUDA_BACKEND) || defined(ENABLE_CUDA_OFFLOAD)
#include <cuda_runtime.h>
#endif

namespace milhoja {

// According to OpenACC v2.7, valid queues are indexed by
//  * non-negative integers, 
//  * acc_async_noval (negative), or
//  * acc_async_sync (negative).
// A Stream object will only be meaningful if it gives access to an 
// actual stream/queue.  Therefore, any negative queue ID should be
// interpreted as invalid.
constexpr int NULL_ACC_ASYNC_QUEUE = -1;

struct Stream {
#if defined(MILHOJA_USE_CUDA_BACKEND) || defined(ENABLE_CUDA_OFFLOAD)
    // cudaStream_t is a typedef of a pointer.  Therefore, this
    // default value is valid and moves should be quick.
    cudaStream_t   cudaStream = nullptr;
#endif
#if defined(MILHOJA_ENABLE_OPENACC_OFFLOAD)
    int            accAsyncQueue = NULL_ACC_ASYNC_QUEUE;
#endif

    Stream(void) = default;
    ~Stream(void)  { };

    Stream(Stream&& stream) {
#if defined(MILHOJA_USE_CUDA_BACKEND) || defined(ENABLE_CUDA_OFFLOAD)
        cudaStream = stream.cudaStream;
        stream.cudaStream = nullptr;
#endif
#if defined(MILHOJA_ENABLE_OPENACC_OFFLOAD)
        accAsyncQueue = stream.accAsyncQueue;
        stream.accAsyncQueue = NULL_ACC_ASYNC_QUEUE; 
#endif
    }

    Stream& operator=(Stream&& rhs) {
#if defined(MILHOJA_USE_CUDA_BACKEND) || defined(ENABLE_CUDA_OFFLOAD)
        cudaStream = rhs.cudaStream;
        rhs.cudaStream = nullptr;
#endif
#if defined(MILHOJA_ENABLE_OPENACC_OFFLOAD)
        accAsyncQueue = rhs.accAsyncQueue;
        rhs.accAsyncQueue = NULL_ACC_ASYNC_QUEUE; 
#endif

        return *this;
    }

    Stream(Stream&)                  = delete;
    Stream(const Stream&)            = delete;
    Stream& operator=(Stream&)       = delete;
    Stream& operator=(const Stream&) = delete;

    bool   isValid(void) const {
#if defined(MILHOJA_USE_CUDA_BACKEND) || defined(ENABLE_CUDA_OFFLOAD)
        if (cudaStream == nullptr) {
            return false;
        }
#endif
#if defined(MILHOJA_ENABLE_OPENACC_OFFLOAD)
        if (accAsyncQueue == NULL_ACC_ASYNC_QUEUE) {
            return false;
        }
#endif
        // This specifies that a Stream object used in a simulation that does
        // not use Streams will alway be valid.  This might be incorrect in the
        // future.
        return true;
    }
};

}
#endif

