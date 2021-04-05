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

namespace orchestration {

enum class PacketDataLocation {NOT_ASSIGNED, CC1, CC2};

struct PacketContents {
    unsigned int            level      = 0;
    std::shared_ptr<Tile>   tileDesc_h = std::shared_ptr<Tile>{};
    Real*                   CC1_data_p = nullptr;
    Real*                   CC2_data_p = nullptr;
    RealVect*               deltas_d   = nullptr;
    IntVect*                lo_d       = nullptr;
    IntVect*                hi_d       = nullptr;
    IntVect*                loGC_d     = nullptr;
    IntVect*                hiGC_d     = nullptr;
    FArray1D*               xCoords_d  = nullptr;  //!< From loGC to hiGC
    FArray1D*               yCoords_d  = nullptr;  //!< From loGC to hiGC
    FArray1D*               zCoords_d  = nullptr;  //!< From loGC to hiGC
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
    static std::unique_ptr<DataPacket>   createPacket(void);

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
    void*                  pointerToStart_host(void) { return packet_p_; };
    void*                  pointerToStart_gpu(void)  { return packet_d_; };
    std::size_t            sizeInBytes(void) const   { return nBytesPerPacket_; }
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
    virtual Real*                 timeStepGpu(void) const = 0;

protected:
    DataPacket(void);

    void         nullify(void);
    std::string  isNull(void) const;

    PacketDataLocation                     location_;
    void*                                  packet_p_;
    void*                                  packet_d_;
    std::deque<std::shared_ptr<Tile>>      tiles_;
    std::size_t*                           nTiles_d_;
    PacketContents*                        contents_p_;
    PacketContents*                        contents_d_;
    Stream                                 stream_;
    std::size_t                            nBytesPerPacket_;

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
    int    startVariable_;
    int    endVariable_;

};

}

#endif

