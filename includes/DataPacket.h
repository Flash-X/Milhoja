#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#if defined(ENABLE_CUDA_OFFLOAD) || defined(USE_CUDA_BACKEND)
#include <cuda_runtime.h>
#endif

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Tile.h"
#include "DataItem.h"
#include "Stream.h"

// TODO: What if we have a DataPacket that starts at the host, is transferred to
// the GPU, then to the FPGA, and finally back to the host?  In this sense,
// no single ThreadTeam would be associated with this class.  Therefore, it
// appears that there should just be a single DataPacket of blocks and we should
// be able to populate it with the pointers that it needs for its full trip.
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

namespace orchestration {

enum class PacketDataLocation {NOT_ASSIGNED, CC1, CC2};

// TODO: This structure makes unpacking of the data packet trivial in the patch
// code.  This was necessary since a humble human was writing and maintaining
// the code.  The code generator, however, will hopefully be able to write the
// low-level unpacking code directly in the patch code.  Therefore, this
// structure, should disappear.  This makes sense as otherwise, this struct
// would have to include every possible tile-specific element that could ever be
// included in a data packet, which is somewhat out of control of this library.
struct PacketContents {
    unsigned int            level      = 0;
    std::shared_ptr<Tile>   tileDesc_h = std::shared_ptr<Tile>{};
    RealVect*               deltas_d   = nullptr;
    IntVect*                lo_d       = nullptr;
    IntVect*                hi_d       = nullptr;
    FArray4D*               CC1_d      = nullptr;  //!< From loGC to hiGC
    FArray4D*               CC2_d      = nullptr;  //!< From loGC to hiGC   
    FArray4D*               FCX_d      = nullptr;  //!< From lo to hi 
    FArray4D*               FCY_d      = nullptr;  //!< From lo to hi  
    FArray4D*               FCZ_d      = nullptr;  //!< From lo to hi  
};

/**
 * \todo initiateDeviceToHost is CUDA specific.  Can we use preprocessor to
 * allow for each backend to have its own version?
 *
 */
class DataPacket : public DataItem {
public:
    virtual std::unique_ptr<DataPacket>  clone(void) const = 0;

    virtual ~DataPacket(void);

    DataPacket(DataPacket&)                  = delete;
    DataPacket(const DataPacket&)            = delete;
    DataPacket(DataPacket&& packet)          = delete;
    DataPacket& operator=(DataPacket&)       = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&)      = delete;

    std::size_t            nTiles(void) const        { return tiles_.size(); }
    const std::size_t*     nTilesGpu(void) const     { return nTiles_d_; }
    void                   addTile(std::shared_ptr<Tile>&& tileDesc);
    std::shared_ptr<Tile>  popTile(void);
    const PacketContents*  tilePointers(void) const  { return contents_d_; };

    virtual void           pack(void) = 0;
    void*                  copyToGpuStart_host(void)           { return (void*)copyInStart_p_; };
    void*                  copyToGpuStart_gpu(void)            { return (void*)copyInStart_d_; };
    std::size_t            copyToGpuSizeInBytes(void) const    { return nCopyToGpuBytes_; }
    void*                  returnToHostStart_host(void)        { return (void*)copyInOutStart_p_; };
    void*                  returnToHostStart_gpu(void)         { return (void*)copyInOutStart_d_; };
    std::size_t            returnToHostSizeInBytes(void) const { return nReturnToHostBytes_; }
    void                   unpack(void);

#ifdef ENABLE_OPENACC_OFFLOAD
    int                    asynchronousQueue(void) { return stream_.accAsyncQueue; }
#endif
#if defined(ENABLE_CUDA_OFFLOAD) || defined(USE_CUDA_BACKEND)
    cudaStream_t           stream(void)            { return stream_.cudaStream; };
#endif

    PacketDataLocation    getDataLocation(void) const;
    void                  setDataLocation(const PacketDataLocation location);
    void                  setVariableMask(const int startVariable, 
                                          const int endVariable);

    // FIXME:  This is a temporary solution as it seems like the easiest way to
    // get dt in to the GPU memory.  There is no reason for dt to be included in
    // each data packet.  It is a "global" value that is valid for all blocks
    // and all operations applied during a solution advance phase.
    virtual Real*         timeStepGpu(void) const = 0;

protected:
    DataPacket(void);

    void         nullify(void);
    std::string  isNull(void) const;

    struct BlockPointersPinned {
        Real*    CC1_data = nullptr;
        Real*    CC2_data = nullptr;
    };

    PacketDataLocation                     location_;
    void*                                  packet_p_;
    void*                                  packet_d_;
    char*                                  copyInStart_p_;
    char*                                  copyInStart_d_;
    char*                                  copyInOutStart_p_;
    char*                                  copyInOutStart_d_;
    std::deque<std::shared_ptr<Tile>>      tiles_;
    std::size_t*                           nTiles_d_;
    PacketContents*                        contents_p_;
    PacketContents*                        contents_d_;
    BlockPointersPinned*                   pinnedPtrs_;
    Stream                                 stream_;
    std::size_t                            nCopyToGpuBytes_;
    std::size_t                            nReturnToHostBytes_;

    static constexpr std::size_t    N_ELEMENTS_PER_CC_PER_VARIABLE =   (NXB + 2 * NGUARD * K1D)
                                                                     * (NYB + 2 * NGUARD * K2D)
                                                                     * (NZB + 2 * NGUARD * K3D);
    static constexpr std::size_t    N_ELEMENTS_PER_CC  = N_ELEMENTS_PER_CC_PER_VARIABLE * NUNKVAR;

    static constexpr std::size_t    N_ELEMENTS_PER_FCX_PER_VARIABLE = (NXB + 1) * NYB * NZB;
    static constexpr std::size_t    N_ELEMENTS_PER_FCX = N_ELEMENTS_PER_FCX_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    N_ELEMENTS_PER_FCY_PER_VARIABLE = NXB * (NYB + 1) * NZB;
    static constexpr std::size_t    N_ELEMENTS_PER_FCY = N_ELEMENTS_PER_FCY_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    N_ELEMENTS_PER_FCZ_PER_VARIABLE = NXB * NYB * (NZB + 1);
    static constexpr std::size_t    N_ELEMENTS_PER_FCZ = N_ELEMENTS_PER_FCZ_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    DRIVER_DT_SIZE_BYTES =          sizeof(Real);
    static constexpr std::size_t    DELTA_SIZE_BYTES     =          sizeof(RealVect);
    static constexpr std::size_t    CC_BLOCK_SIZE_BYTES  = N_ELEMENTS_PER_CC
                                                                  * sizeof(Real);
    static constexpr std::size_t    FCX_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCX
                                                                  * sizeof(Real);
    static constexpr std::size_t    FCY_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCY
                                                                  * sizeof(Real);
    static constexpr std::size_t    FCZ_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCZ
                                                                  * sizeof(Real);
    static constexpr std::size_t    POINT_SIZE_BYTES     =          sizeof(IntVect);
    static constexpr std::size_t    ARRAY1_SIZE_BYTES    =          sizeof(FArray1D);
    static constexpr std::size_t    ARRAY4_SIZE_BYTES    =          sizeof(FArray4D);
    static constexpr std::size_t    COORDS_X_SIZE_BYTES  = (NXB + 2 * NGUARD * K1D)
                                                                  * sizeof(Real);
    static constexpr std::size_t    COORDS_Y_SIZE_BYTES  = (NYB + 2 * NGUARD * K2D)
                                                                  * sizeof(Real);
    static constexpr std::size_t    COORDS_Z_SIZE_BYTES  = (NZB + 2 * NGUARD * K3D)
                                                                  * sizeof(Real);

private:
    int   startVariable_;
    int   endVariable_;
};

}

#endif

