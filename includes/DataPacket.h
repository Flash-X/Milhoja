#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#include <stdexcept>

#if defined(ENABLE_CUDA_OFFLOAD) || defined(USE_CUDA_BACKEND)
#include <cuda_runtime.h>
#endif

#include "DataItem.h"
#include "DataShape.h"
#include "Stream.h"
#include "Tile.h"
#include <deque>

namespace orchestration {

/**
 * @class DataPacket DataPacket.h
 * @brief Define and effectively export to applications the minimal DataPacket
 * interface that their concrete DataPackets must implement.
 * @todo Many if not all of the static constexpr lines should be moved to a file
 *       such as constants.h.
 */
class DataPacket : public DataItem {
protected:
    /****** STREAMS/QUEUES ******/
    Stream                            mainStream_;    //!< Main stream for communication
    std::deque<Stream>                extraStreams_;  //!< Additional streams for computation

    /****** ITEMS ******/
    std::deque<std::shared_ptr<Tile>> items_;         //!< Items included in packet
    std::deque<DataShape>             itemShapes_;    //!< Shapes corresponding to individual items

    /****** MEMORY ******/
    enum MemoryPartition {SHARED, IN, INOUT, OUT, SCRATCH, _N};
private:
    std::size_t memorySize_[MemoryPartition::_N];     //!< Bytes of memory for each partition
    void*       memoryPtr_src_;                       //!< Location of source memory
    void*       memoryPtr_trg_;                       //!< Location in destination memory

public:
    virtual ~DataPacket(void);

    DataPacket(DataPacket&)                  = delete;
    DataPacket(const DataPacket&)            = delete;
    DataPacket(DataPacket&& packet)          = delete;
    DataPacket& operator=(DataPacket&)       = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&)      = delete;

    /**
     * Obtain a pointer to a new, empty DataPacket of the same concrete type as
     * the calling object.  The main workhorse of the Prototype design pattern.
     *
     * @return The pointer.  For the main use cases in the runtime, this should
     * be cast to a shared_ptr.  We return a unique_ptr based on the discussion
     * in Item 19 (Pp 113) of Effective Modern C++.
     *
     * @todo Add in citation.
     */
    virtual std::unique_ptr<DataPacket> clone(void) const = 0;

    /**
     * The runtime calls this member function automatically once all tiles to be
     * included in the DataPacket have been added using the addTile() function.
     * Derived classes implement this functionality and in doing so specify the
     * structure of the data packet and pack the DataPacket's content in host
     * memory in preparation for transfer to device memory.  Concrete
     * implementations determine in what type of memory packing occurs and to
     * what type of memory the data is sent.
     */
    virtual void  pack(void) = 0;

    virtual void  unpack(void) = 0;

    /****** STREAMS/QUEUES ******/

#if defined(ENABLE_OPENACC_OFFLOAD)
    /**
     * Obtain the main OpenACC asynchronous queue assigned to the packet, on
     * which communications are scheduled and computation can also be scheduled.
     * This can be called after pack() is called and before unpack() is called.
     */
    int           mainAsyncQueue(void) const { return mainStream_.accAsyncQueue; }

    /**
     * Obtain the indicated extra OpenACC asynchronous queue assigned to the
     * packet for concurrent kernel execution.  This can be called after pack()
     * is called and before unpack() is called.  These queues must be explicitly
     * released by the task function using releaseExtraQueue() and before the
     * task function terminates.
     *
     * While calling code can call this as many times as desired for any given
     * id, it is a logical error to call this function to obtain a function that has
     * already been released.
     *
     * It is the task function's responsibility to use this queue correctly.
     * This includes synchronizing computation on this queue with the arrival of
     * data on the packet's main queue.
     *
     * @param id - the index of the queue to obtain.  If the packet has acquired
     * N total queues, valid values are 2 to N inclusive.
     */
    int           extraAsyncQueue(unsigned int id) { return extraStreams_.at(id).accAsyncQueue; }

    unsigned int  nExtraAsyncQueues(void) { return extraStreams_.size(); }
#endif /* defined(ENABLE_OPENACC_OFFLOAD) */
#if defined(ENABLE_CUDA_OFFLOAD) || defined(USE_CUDA_BACKEND)
    /**
     * Obtain the main CUDA stream assigned to the DataPacket for transferring the
     * packet to and from the host.  This can be called after pack() is called
     * and before unpack() is called.  Calling code can call this as many times
     * as desired.
     */
    cudaStream_t  mainStream(void) const { return mainStream_.cudaStream; };

    cudaStream_t  extraStream(unsigned int id) { return extraStreams_.at(id).cudaStream; }

    unsigned int  nExtraStreams(void) { return extraStreams_.size(); }
#endif /* defined(ENABLE_CUDA_OFFLOAD) || defined(USE_CUDA_BACKEND) */

    /****** ITEMS ******/

    /**
     * Add new item to the data packet. Use default data shape or specify a shape.
     */
    void                      addTile(std::shared_ptr<Tile>&& item);

    /**
     * Get next item from the data packet.
     */
    std::shared_ptr<Tile>     popTile(void);

    /**
     * Obtain the number of items presently included in the packet.
     */
    unsigned int              nTiles(void) const { return items_.size(); }

    /****** MEMORY ******/

    std::size_t               getInSize(void) const {
        assert(hasValidMemorySizes());
        return memorySize_[MemoryPartition::SHARED] +
               memorySize_[MemoryPartition::IN] +
               memorySize_[MemoryPartition::INOUT];
    }

    std::size_t               getOutSize(void) const {
        assert(hasValidMemorySizes());
        return memorySize_[MemoryPartition::INOUT] +
               memorySize_[MemoryPartition::OUT];
    }

    /**
     * Obtain a pointer to the start of the contiguous block of memory on the
     * host side that will be transferred to GPU memory.
     */
    void*                     getInPointer_src(void)  const { return memoryPtr_src_; }

    /**
     * Obtain a pointer to the start of the contiguous block of memory in GPU
     * device memory that will receive the packet's data from the host.
     */
    void*                     getInPointer_trg(void)  const { return memoryPtr_trg_; }

    /**
     * Obtain a pointer to the start of the contiguous block of memory in host
     * memory that will receive the packet's data upon transfer back to the
     * host.
     */
    void*                     getOutPointer_src(void) const { return memoryPtr_src_; }

    /**
     * Obtain a pointer to the start of the contiguous block of memory on the
     * GPU side that will be transferred back to host memory.
     */
    void*                     getOutPointer_trg(void) const {
        std::size_t offset = memorySize_[MemoryPartition::SHARED] + memorySize_[MemoryPartition::IN];

        assert(hasValidMemorySizes());
        static_assert(sizeof(char) == 1, "Invalid char size");
        return (void*)( ((char*)memoryPtr_trg_) + offset );
    }

protected:
    /**
     * Constructor
     */
    DataPacket(void);

    /**
     *
     */
    void          pack_initialize(void);
    void          pack_finalize(void);
    void          pack_finalize(unsigned int);

    /**
     *
     */
    void          unpack_initialize(void);
    void          unpack_finalize(void);

    /**
     *
     */
    void          extraStreams_request(unsigned int);

    /**
     * Release the indicated extra OpenACC asynchronous queue.  This must be
     * called after calling pack() and before calling unpack().  It is a logical
     * error to call this more than once for any given id or on the main queue.
     *
     * @param id - the index of the queue to release.
     */
    void          extraStreams_release(void);

    /**
     *
     */
    void          extraStreams_releaseId(unsigned int);

    /**
     *
     */
    virtual void  setupItemShapes(void) = 0;

    /**
     *
     */
    virtual void  clearItemShapes(void) = 0;

    bool          isUniformItemShape(void) const {
        return (1 != items_.size() && 1 == itemShapes_.size());
    }

    const DataShape&    getItemShape(unsigned int itemId) const {
        assert(0 < itemShapes_.size());
        if (isUniformItemShape()) {
            return itemShapes_.at(0);
        } else {
            return itemShapes_.at(itemId);
        }
    }

    /**
     *
     */
    void*         getInPointerToPartItemVar_src(MemoryPartition, unsigned int, unsigned int) const;
    void*         getOutPointerToPartItemVar_src(MemoryPartition, unsigned int, unsigned int) const;
    void*         getPointerToPartItemVar_trg(MemoryPartition, unsigned int, unsigned int) const;

private:
    void          nullify(void);
    std::string   isNull(void) const;
    void          checkErrorNull(const std::string) const;

    void          mainStream_request(void);
    void          mainStream_release(void);

    void          setupMemorySizes(void);
    void          clearMemorySizes(void);
    bool          hasValidMemorySizes() const;

    std::size_t   getSize_src(void) const;
    std::size_t   getSize_trg(void) const;
    std::size_t   getOffsetToPartItemVar(MemoryPartition, unsigned int, unsigned int) const;
};

}

#endif

