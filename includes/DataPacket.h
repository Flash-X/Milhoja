#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#ifdef ENABLE_CUDA_OFFLOAD
#include <cuda_runtime.h>
#endif

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Tile.h"
#include "DataItem.h"

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
    FArray4D*               CC1_d      = nullptr;
    FArray4D*               CC2_d      = nullptr;
};

class DataPacket : public DataItem {
public:
    static std::unique_ptr<DataPacket>   createPacket(void);

    virtual ~DataPacket(void)  { };

    DataPacket(DataPacket&)                  = delete;
    DataPacket(const DataPacket&)            = delete;
    DataPacket(DataPacket&& packet)          = delete;
    DataPacket& operator=(DataPacket&)       = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&)      = delete;

    virtual std::size_t            nTiles(void) const = 0;
    virtual const std::size_t*     nTilesGpu(void) const = 0;
    virtual void                   addTile(std::shared_ptr<Tile>&& tileDesc) = 0;
    virtual std::shared_ptr<Tile>  popTile(void) = 0;
    virtual const PacketContents*  tilePointers(void) const = 0;

    virtual void                   initiateHostToDeviceTransfer(void) = 0;
    virtual void                   transferFromDeviceToHost(void) = 0;
#ifdef ENABLE_OPENACC_OFFLOAD
    virtual int                    asynchronousQueue(void) = 0;
#endif
#ifdef ENABLE_CUDA_OFFLOAD
    virtual cudaStream_t           stream(void) = 0;
#endif

    virtual PacketDataLocation    getDataLocation(void) const = 0;
    virtual void                  setDataLocation(const PacketDataLocation location) = 0;
    virtual void                  setVariableMask(const int startVariable, 
                                                  const int endVariable) = 0;
protected:
    DataPacket(void)   { };

    static constexpr std::size_t    N_ELEMENTS_PER_BLOCK_PER_VARIABLE =   (NXB + 2 * NGUARD * K1D)
                                                                        * (NYB + 2 * NGUARD * K2D)
                                                                        * (NZB + 2 * NGUARD * K3D);
    static constexpr std::size_t    N_ELEMENTS_PER_BLOCK = N_ELEMENTS_PER_BLOCK_PER_VARIABLE * NUNKVAR;

    static constexpr std::size_t    DELTA_SIZE_BYTES    =           sizeof(RealVect);
    static constexpr std::size_t    BLOCK_SIZE_BYTES    = N_ELEMENTS_PER_BLOCK 
                                                                  * sizeof(Real);
    static constexpr std::size_t    POINT_SIZE_BYTES    =           sizeof(IntVect);
    static constexpr std::size_t    ARRAY1_SIZE_BYTES   =           sizeof(FArray1D);
    static constexpr std::size_t    ARRAY4_SIZE_BYTES   =           sizeof(FArray4D);
    static constexpr std::size_t    COORDS_X_SIZE_BYTES = (NXB + 2 * NGUARD * K1D)
                                                                  * sizeof(Real);
    static constexpr std::size_t    COORDS_Y_SIZE_BYTES = (NYB + 2 * NGUARD * K2D)
                                                                  * sizeof(Real);
    static constexpr std::size_t    COORDS_Z_SIZE_BYTES = (NZB + 2 * NGUARD * K3D)
                                                                  * sizeof(Real);
};

}

#endif

