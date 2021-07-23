#ifndef DATA_PACKET_HYDRO_GPU_3_H__
#define DATA_PACKET_HYDRO_GPU_3_H__

#include "DataPacket.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "Grid_REAL.h"
#include "FArray1D.h"
#include "FArray4D.h"

namespace orchestration {

class DataPacket_Hydro_gpu_4 : public DataPacket {
public:
    struct ItemContext {
        std::shared_ptr<Tile>   tileDesc_h;
        RealVect*               deltas_d;
        IntVect*                lo_d;
        IntVect*                hi_d;
        FArray4D                UNK_d;  //!< From loGC to hiGC
        FArray4D                AUX_d;  //!< From loGC to hiGC
        FArray4D                FCX_d;  //!< From lo to hi
#if NDIM >= 2
        FArray4D                FCY_d;  //!< From lo to hi
#endif
#if NDIM == 3
        FArray4D                FCZ_d;  //!< From lo to hi
#endif
    };

    struct SharedContext {
        Real          dt;
        unsigned int  nItems;
        ItemContext*  itemCtx;
    };

private:
    enum Variable {
        SCTX,   // shared context
        ICTX,   // item context
        DELTAS, // deltas
        LO,     // lo-indices
        HI,     // hi-indices
        UNK,    // unknowns
        AUX,    // auxiliary
        FCX,    // x-fluxes
#if NDIM >= 2
        FCY,    // y-fluxes
#endif
#if NDIM == 3
        FCZ,    // z-fluxes
#endif
        _N
    };

    static constexpr unsigned int SHAPE_DIM = 5;
    static constexpr std::size_t  VARIABLE_SHAPES[Variable::_N * SHAPE_DIM] = {
        /* SCTX  */  sizeof(SharedContext), 1, 0, 0, 0,
        /* ICTX  */  0,                     0, 0, 0, 0,  // zero, because #items is unknown a-priori
        /* DELTAS */ sizeof(RealVect),      1, 0, 0, 0,
        /* LO    */  sizeof(IntVect),       1, 0, 0, 0,
        /* HI    */  sizeof(IntVect),       1, 0, 0, 0,
        /* UNK   */  sizeof(Real),          NXB + 2*NGUARD*K1D, NYB + 2*NGUARD*K2D, NZB + 2*NGUARD*K3D, NUNKVAR - 1,
        /* AUX   */  sizeof(Real),          NXB + 2*NGUARD*K1D, NYB + 2*NGUARD*K2D, NZB + 2*NGUARD*K3D, 1,
        /* FCX   */  sizeof(Real),          (NXB + 1), NYB, NZB, NFLUXES,
#if NDIM >= 2
        /* FCY   */  sizeof(Real),          NXB, (NYB + 1), NZB, NFLUXES,
#endif
#if NDIM == 3
        /* FCZ   */  sizeof(Real),          NXB, NYB, (NZB + 1), NFLUXES,
#endif
    };

    static constexpr unsigned int VARIABLE_PARTITIONS[Variable::_N] = {
        /* SCTX  */  MemoryPartition::SHARED,
        /* ICTX  */  MemoryPartition::SHARED,
        /* DELTAS */ MemoryPartition::IN,
        /* LO    */  MemoryPartition::IN,
        /* HI    */  MemoryPartition::IN,
        /* UNK   */  MemoryPartition::INOUT,
        /* AUX   */  MemoryPartition::SCRATCH,
        /* FCX   */  MemoryPartition::SCRATCH,
#if NDIM >= 2
        /* FCY   */  MemoryPartition::SCRATCH,
#endif
#if NDIM == 3
        /* FCZ   */  MemoryPartition::SCRATCH,
#endif
    };

    //std::deque<DataMask> itemMasks_; //!< Masks corresponding to individual tiles

public:
    DataPacket_Hydro_gpu_4(void);
    ~DataPacket_Hydro_gpu_4(void);

    DataPacket_Hydro_gpu_4(DataPacket_Hydro_gpu_4&)                  = delete;
    DataPacket_Hydro_gpu_4(const DataPacket_Hydro_gpu_4&)            = delete;
    DataPacket_Hydro_gpu_4(DataPacket_Hydro_gpu_4&& packet)          = delete;
    DataPacket_Hydro_gpu_4& operator=(DataPacket_Hydro_gpu_4&)       = delete;
    DataPacket_Hydro_gpu_4& operator=(const DataPacket_Hydro_gpu_4&) = delete;
    DataPacket_Hydro_gpu_4& operator=(DataPacket_Hydro_gpu_4&& rhs)  = delete;

    std::unique_ptr<DataPacket> clone(void) const override;
    void                        pack(void) override;
    void                        unpack(void) override;

    SharedContext*              getSharedContext(void) const {
        return static_cast<SharedContext*>(getPointerToPartItemVar_trg(MemoryPartition::SHARED, 0, Variable::SCTX));
    }

protected:
    void                        setupItemShapes(void) override;
    void                        clearItemShapes(void) override;
};

}

#endif

