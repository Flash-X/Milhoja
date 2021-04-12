/**
 * \file   Backend.h
 *
 * A class that declares the generic interface for all runtime functionality
 * that must be implemented on a per-platform, per-vendor, or per-device basis.
 * Each concrete implementation of this functionality is to be implemented in a
 * class derived from this class.
 *
 * Some backend functionality is related to acquiring and releasing shared
 * resoures such as streams and memory.  So that the backends can effectively
 * manage these resources, it is important that all code that uses the runtime
 * aquire all such resources through the backend rather than doing so directly
 * on their own.  In particular, physics codes run by the runtime should *not*
 * allocate their own memory directly, but rather request that the runtime
 * acquire memory on its behalf and pass it as an argument to the physics code.
 *
 * This class is designed using a adaption of the the Singleton design
 * pattern presented in CITATION NEEDED HERE.  This means that the runtime
 * accesses the desired functionality using polymorphism.  As such, there is
 * little sense in concrete backends extending the public Backend interface.
 *
 * The choice of which particular version of the backend will implement this
 * interface is given at setup time and propagated to this class via a
 * compilation flag.  All code that needs backend functionality should acquire
 * the backend instance through the instance function in this class and should
 * never use directly one of the derived classes.
 *
 * Member functions of this class make reference to streams, which is a term
 * often associated with CUDA.  The runtime implements its own Stream class,
 * which is a more generic representation of a CUDA stream.  The word stream in
 * this interface refers to this custom class.
 *
 * @todo Figure out how to use Doxygen's cite keyword here.
 */

#ifndef BACKEND_H__
#define BACKEND_H__

#include <cstddef>

#ifdef USE_CUDA_BACKEND
#include <cuda_runtime.h>
#endif

#include "Stream.h"
#include "DataPacket.h"

namespace orchestration {

#ifdef USE_CUDA_BACKEND
using GPU_TO_HOST_CALLBACK_FCN = cudaHostFn_t;
#else
#error "Choose a backend that can use a GPU-to-host callback routine"
#endif

class Backend {
public:
    virtual ~Backend(void)     { instantiated_ = false; };

    Backend(Backend&)                  = delete;
    Backend(const Backend&)            = delete;
    Backend(Backend&&)                 = delete;
    Backend& operator=(Backend&)       = delete;
    Backend& operator=(const Backend&) = delete;
    Backend& operator=(Backend&&)      = delete;

    static void       instantiate(const unsigned int nStreams,
                                  const std::size_t  nBytesInMemoryPools);
    static Backend&   instance(void);

    //----- STREAMS
    /**
     * Obtain the maximum number of streams that the runtime is allowed to use
     * at any given point in time.
     *
     * @return The number of streams
     */
    virtual int       maxNumberStreams(void) const = 0; 

    /**
     * Obtain the number of streams presently available for distribution.
     *
     * @return The number of streams
     */
    virtual int       numberFreeStreams(void) = 0;

    /**
     * Acquire a free stream for exclusive use.  Once acquired, the returned
     * Stream object must eventually be released by calling releaseStream().
     *
     * @param block - If true, then this function will block execution of the
     * calling thread until a stream becomes availble.  If false, then the
     * function is not blocked.
     * @return The requested Stream object.  If block is False and no streams
     * are available, then a null Stream object is returned.
     */
    virtual Stream    requestStream(const bool block) = 0;

    /**
     * Release a Stream object that was acquired using requestStream().  It is a
     * logical error to release a stream that was not acquired with
     * requestStream, a null stream, or a stream that has already been released.
     *
     * @param stream - the Stream object to release
     */
    virtual void      releaseStream(Stream& stream) = 0;

    //----- DATA MOVEMENT
    /**
     * Initiate an asynchronous host-to-GPU transfer of the given data packet.
     * This function should only block for the amount of time it takes to initiate
     * the transfer.
     *
     * @param packet - the data packet to transfer.
     */
    virtual void      initiateHostToGpuTransfer(DataPacket& packet) = 0;

    /**
     * Initiate an asynchronous GPU-to-host transfer of the given data packet
     * and define what callback function should be called once the transfer
     * finishes.  This function should only block for the amount of time it
     * takes to initiate the transfer.
     *
     * @param packet - the data packet to transfer
     * @param callback - the callback rountine that will be called
     * automatically once the transfer completes.
     * @param callbackData - a pointer to the data needed by the callback
     * routine to finalize the transer.
     */
    virtual void      initiateGpuToHostTransfer(DataPacket& packet,
                                                GPU_TO_HOST_CALLBACK_FCN callback,
                                                void* callbackData) = 0;

    //----- MEMORY MANAGEMENT
    /**
     * To facilitate the construction of data packets, pinned and GPU memory can
     * be acquired in one call.  The given memory blocks are for exclusive use
     * by the calling code.  Once finished with the memory, the calling code
     * must release both memory blocks in a single call to releaseGpuMemory.
     *
     * @todo Define the desired behavior of this function when there are
     * insufficient memory resources.  Block?  Should we have a flag like
     * requestStream has?
     *
     * @param pinnnedBytes - the number of contiguous bytes to be acquired in
     * pinned memory
     * @param pinnedPtr - the location of the start of the contiguous pinned
     * memory block obtained
     * @param gpuBytes - the number of contiguous bytes to be acquired in
     * GPU memory
     * @param gpuPtr - the location of the start of the contiguous GPU
     * memory block obtained
     */
    virtual void      requestGpuMemory(const std::size_t pinnedBytes,
                                       void** pinnedPtr,
                                       const std::size_t gpuBytes,
                                       void** gpuPtr) = 0;

    /**
     * Release a contiguous region of pinned memory and a contiguous region of
     * GPU memory that were previously acquired with releaseGpuMemory.  It is a
     * logical error to release memory that was not acquired with
     * requestGpuMemory or memory that has already been released.
     *
     * @todo Can users acquire pinned and gpu memory separately, but release in
     * one call to this function?  What about acquiring in a single call and
     * releasing in separate calls?  Acquiring pinned/GPU memory in two calls and
     * releasing the pinned from one acquisition with the GPU of another
     * acquisition?
     *
     * @param pinnedPtr - the location of the start of the contiguous pinned
     * memory block to release.  This will be nullified.
     * @param gpuPtr - the location of the start of the contiguous GPU
     * memory block to release.  This will be nullified.
     */
    virtual void      releaseGpuMemory(void** pinnedPtr, void** gpuPtr) = 0;

    /**
     * Alert the runtime backend that an invocation of the runtime has ceased
     * and that its memory accounting can be reset in preparation for the next
     * invocation.
     *
     * @todo Remove this once a real memory manager is implemented.
     */
    virtual void      reset(void) = 0;

protected:
    Backend(void)     {};

private:
    static bool           instantiated_;
    static unsigned int   nStreams_;
    static std::size_t    nBytesInMemoryPools_;
};

}

#endif

