#ifndef CUDA_DATA_PACKET_H__
#define CUDA_DATA_PACKET_H__

#include <deque>
#include <memory>

#include "Tile.h"
#include "DataPacket.h"
#include "Stream.h"

namespace orchestration {

// TODO: Will the offline toolchain write this?  If so, could we have templates that
// just need tweaking?
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
class CudaDataPacket : public DataPacket {
public:
    CudaDataPacket(void);
    ~CudaDataPacket(void);

    CudaDataPacket(CudaDataPacket&)                  = delete;
    CudaDataPacket(const CudaDataPacket&)            = delete;
    CudaDataPacket(CudaDataPacket&& packet)          = delete;
    CudaDataPacket& operator=(CudaDataPacket&)       = delete;
    CudaDataPacket& operator=(const CudaDataPacket&) = delete;
    CudaDataPacket& operator=(CudaDataPacket&& rhs)  = delete;

    // Overrides of DataPacket member functions
    std::size_t            nTiles(void) const override;
    const std::size_t*     nTilesGpu(void) const override { return nTiles_d_; };
    void                   addTile(std::shared_ptr<Tile>&& tileDesc) override;
    std::shared_ptr<Tile>  popTile(void) override;
    const PacketContents*  tilePointers(void) const override { return contents_d_; };

    void                   initiateHostToDeviceTransfer(void) override;
    void                   initiateDeviceToHostTransfer(cudaHostFn_t callback,
                                                        void* callbackData) override;
    void                   unpack(void) override;
#ifdef ENABLE_OPENACC_OFFLOAD
    int                    asynchronousQueue(void) override { return stream_.accAsyncQueue; }
    Stream&                extraStream(const int idx)       { return streamsExtra_[idx]; }
#endif
#ifdef ENABLE_CUDA_OFFLOAD
    cudaStream_t           stream(void) override { return stream_.cudaStream; };
#endif

    PacketDataLocation     getDataLocation(void) const override;
    void                   setDataLocation(const PacketDataLocation location) override;
    void                   setVariableMask(const int startVariable, 
                                           const int endVariable) override;

    Real*                  timeStepGpu(void) const override  { return dt_d_; }

protected:
    // Fix to one block per data packet as first step but with a scratch block
    static constexpr std::size_t    N_BLOCKS = 2; 

    void         nullify(void);
    std::string  isNull(void) const;

    void         pack(void);

private:
    PacketDataLocation                     location_;
    int                                    startVariable_;
    int                                    endVariable_;
    void*                                  packet_p_;
    void*                                  packet_d_;
    std::deque<std::shared_ptr<Tile>>      tiles_;
    std::size_t*                           nTiles_d_;
    PacketContents*                        contents_p_;
    PacketContents*                        contents_d_;
    Stream                                 stream_;
    Stream*                                streamsExtra_;
    std::size_t                            nBytesPerPacket_;

    Real*                                  dt_d_;
};

}

#endif

