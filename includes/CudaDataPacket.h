#ifndef CUDA_DATA_PACKET_H__
#define CUDA_DATA_PACKET_H__

#include <memory>

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Tile.h"
#include "DataItem.h"
#include "CudaStream.h"
#include "PacketDataLocation.h"

#include "constants.h"
#include "Flash.h"

namespace orchestration {

// TODO: This DataPacket is linked to CUDA.  Is this acceptable?
// Will the offline toolchain write this?  If so, could we have templates that
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
class CudaDataPacket : public DataItem {
public:
    struct Contents {
        unsigned int  level   = 0;
        RealVect*     deltas  = nullptr;
        IntVect*      lo      = nullptr;
        IntVect*      hi      = nullptr;
        IntVect*      loGC    = nullptr;
        IntVect*      hiGC    = nullptr;
        FArray1D*     xCoords = nullptr;
        FArray1D*     yCoords = nullptr;
        FArray4D*     CC1     = nullptr;
        FArray4D*     CC2     = nullptr;
    };

    CudaDataPacket(std::shared_ptr<Tile>&& tileDesc);
    ~CudaDataPacket(void);

    CudaDataPacket(CudaDataPacket&)                  = delete;
    CudaDataPacket(const CudaDataPacket&)            = delete;
    CudaDataPacket(CudaDataPacket&& packet)          = delete;
    CudaDataPacket& operator=(CudaDataPacket&)       = delete;
    CudaDataPacket& operator=(const CudaDataPacket&) = delete;
    CudaDataPacket& operator=(CudaDataPacket&& rhs)  = delete;

    std::shared_ptr<DataItem>  getTile(void) { return tileDesc_; };

    void            initiateHostToDeviceTransfer(void);
    void            transferFromDeviceToHost(void);

    CudaStream&     stream(void)             { return stream_; };
    const Contents  gpuContents(void) const  { return contents_d_; };

    PacketDataLocation    getDataLocation(void) const;
    void                  setDataLocation(const PacketDataLocation location);
    void                  setVariableMask(const int sVar, 
                                          const int eVar);

protected:
    static constexpr std::size_t    N_CELLS_PER_VARIABLE =   (NXB + 2 * NGUARD * K1D)
                                                           * (NYB + 2 * NGUARD * K2D)
                                                           * (NZB + 2 * NGUARD * K3D);
    static constexpr std::size_t    N_CELLS = N_CELLS_PER_VARIABLE * NUNKVAR;

    static constexpr std::size_t    DELTA_SIZE_BYTES    =           sizeof(RealVect);
    static constexpr std::size_t    BLOCK_SIZE_BYTES    = N_CELLS * sizeof(Real);
    static constexpr std::size_t    POINT_SIZE_BYTES    =           sizeof(IntVect);
    static constexpr std::size_t    ARRAY1_SIZE_BYTES   =           sizeof(FArray1D);
    static constexpr std::size_t    ARRAY4_SIZE_BYTES   =           sizeof(FArray4D);
    static constexpr std::size_t    COORDS_X_SIZE_BYTES =     NXB * sizeof(Real);
    static constexpr std::size_t    COORDS_Y_SIZE_BYTES =     NYB * sizeof(Real);
    // Fix to one block per data packet as first step but with a scratch block
    static constexpr std::size_t    N_BLOCKS = 2; 
    static constexpr std::size_t    N_BYTES_PER_PACKET =          1 * DELTA_SIZE_BYTES
                                                         +        4 * POINT_SIZE_BYTES
                                                         + N_BLOCKS * BLOCK_SIZE_BYTES
                                                         +        2 * ARRAY4_SIZE_BYTES
                                                         +        1 * COORDS_X_SIZE_BYTES
                                                         +        1 * COORDS_Y_SIZE_BYTES
                                                         +        2 * ARRAY1_SIZE_BYTES;

    void         nullify(void);
    std::string  isNull(void) const;

    void         pack(void);
    void         unpack(void);

private:
    std::shared_ptr<Tile>   tileDesc_;
    Real*                   CC1_data_p_;
    Real*                   CC2_data_p_;
    PacketDataLocation      location_;
    int                     startVariable_;
    int                     endVariable_;
    void*                   packet_p_;
    void*                   packet_d_;
    Contents                contents_d_;
    CudaStream              stream_;
};

}

#endif

