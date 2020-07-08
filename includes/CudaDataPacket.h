#ifndef CUDA_DATA_PACKET_H__
#define CUDA_DATA_PACKET_H__

#include <vector>

#include "Tile.h"
#include "CudaStream.h"

namespace orchestration {

// TODO: This DataPacket is linked to CUDA.  Is this acceptable?
// Will the offline toolchain write this?  If so, could we have templates that
// just need tweaking?
// TODO: The DataPacket as implemented knows that it exists to transfer between
// host and GPU.  If we accept this, then we would have multiple flavors of
// DataPackets and ThreadTeams *would* be dedicated to particular HW unless we
// can setup the ThreadTeams to work with a common base class of DataPacket.
// TODO: What if we have a DataPacket that starts at the host, is transferred to
// the GPU, then to the FPGA, and finally back to the host?  In this sense,
// no single ThreadTeam would be associated with this class.  Therefore, it
// appears that there should just be a single DataPacket of blocks and we should
// be able to populate it with the pointers that it needs for its full trip.
// TODO:  The packet should have pointers to the different sections of
//        data within the packet.
//        For host->device, send across only the Input and IO section
//        (single continguous block).
//        For device->host, send across only the IO and output section
//        (single contiguous block).
// FIXME: This needs to be written as a full implementation and cleanly.
// TODO: I don't think that the DataPackets need to be thread-safe.  Rather the
// full system design should be such that at any point in time at most one
// thread can have exclusive access to the packet.  In other words, the design
// should enforce the exclusive access without the need for a mutex.  For
// instance, the distributor creates a packet, at which time it has ownership as
// no other thread knows that it exists.  When it enqueues the packet, it
// transfers ownership of the packet to the object on which it enqueued.  If the
// enqueue is a copy, this is accomplished by creating two copies.  The original
// should eventually be transferred over to another entity with a move.  The
// move should leave the original packet null, so that no harm can be done.
// This implies that the distributor should not alter or use the contents of the
// packet after it starts enqueueing and before its last move, which seems
// obvious.  Each entity with a data item queue must be written so that at most
// one thread can access a packet at any point in time.
class CudaDataPacket {
public:
    // How many CC blocks in the DataPacket
    static constexpr std::size_t    N_CC_BUFFERS = 2;
    static constexpr unsigned int   NULL_DATA_PACKET_ID = 0;

    // TODO: Put in facility to create vector with pre-allocated elements?
    // TODO: Add in begin/end so that client code can use for-range?
    CudaDataPacket(void);
    ~CudaDataPacket(void);

    bool         isNull(void) const;
    std::size_t  nDataItems(void) const;
    void         nullify(void);
    void         addDataItem(Tile&& dataItem);
    void         addDataItem(const Tile& dataItem);
    void         prepareForTransfer(const std::size_t nBytesIn);
    // TODO: This name is CUDA specific.  It should be renamed to reflect that
    // it carries out a generic action associated with terminating the lifetime 
    // of a DataPacket.
    // TODO: I think that this should be moved out or the concept of the
    // DataPacket should be expanded so that it handles movements when
    // instructed to do so.
    void         moveDataFromPinnedToSource(void);

    Tile&         operator[](const std::size_t idx);
    const Tile&   operator[](const std::size_t idx) const;

    double*            start_p_;
    void*              copyIn_p_;
    double*            start_d_;
    void*              copyIn_d_;
    CudaStream         stream_;
    std::size_t        nBytes_;

private:
    // Indexing is 1-based so that zero can be used for null packet
    unsigned int       idx_;
    std::size_t        nDataPerBlock_;
    std::vector<Tile>  tileList_;

    // A DataPacket will immediately be enqueued and we therefore need a copy
    // constructor if there are multiple ThreadTeam's on which we will enqueue
    // the packet and a move constructor.
    //
    // The packet could continue to be enqueued with other teams until finally
    // it is time to recover the resources.
    // 
    // TODO: How do we manage the move of tileList_?
    CudaDataPacket(CudaDataPacket&) = delete;
    CudaDataPacket(const CudaDataPacket&) = delete;
    CudaDataPacket(CudaDataPacket&& packet) = delete;
    CudaDataPacket& operator=(CudaDataPacket&) = delete;
    CudaDataPacket& operator=(const CudaDataPacket&) = delete;
    CudaDataPacket& operator=(CudaDataPacket&& rhs) = delete;
};

}

#endif

