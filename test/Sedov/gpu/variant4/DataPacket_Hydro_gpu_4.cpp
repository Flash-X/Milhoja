#include "DataPacket_Hydro_gpu_4.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray4D.h"
#include "Backend.h"

#include "Driver.h"

#include "Flash.h"

#if NFLUXES <= 0
#error "Sedov problem should include fluxes"
#endif

namespace orchestration {

DataPacket_Hydro_gpu_4::DataPacket_Hydro_gpu_4(void)
      : DataPacket{}
        //itemMasks_{}
{
}

DataPacket_Hydro_gpu_4::~DataPacket_Hydro_gpu_4(void) {}

std::unique_ptr<DataPacket>   DataPacket_Hydro_gpu_4::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_Hydro_gpu_4{}};
}

void  DataPacket_Hydro_gpu_4::pack(void) {
    std::size_t   siz;
    unsigned int  nComp;

    // initialize packing
    pack_initialize();

    //----- SHARED MEMORY PARTITION

    SharedContext* sharedCtx_src = static_cast<SharedContext*>(getInPointerToPartItemVar_src(MemoryPartition::SHARED, 0, Variable::ICTX));
    sharedCtx_src->dt            = Driver::dt;
    sharedCtx_src->nItems        = items_.size();
    sharedCtx_src->itemCtx       = static_cast<ItemContext*>(getPointerToPartItemVar_trg(MemoryPartition::SHARED, 0, Variable::ICTX));

    //----- IN, INOUT, SCRATCH MEMORY PARTITIONS

    // get (first) item context
    ItemContext* ctx = static_cast<ItemContext*>(getInPointerToPartItemVar_src(MemoryPartition::SHARED, 0, Variable::ICTX));

    for (unsigned int n=0; n<items_.size(); ++n, ++ctx) { // loop over all items and item contexts
        // get tile information (host)
        Tile*               tileDesc_h = items_[n].get();
        assert(tileDesc_h);
        const RealVect      deltas_h   = tileDesc_h->deltas();
        const IntVect       lo_h       = tileDesc_h->lo();
        const IntVect       hi_h       = tileDesc_h->hi();
        const IntVect       loGC_h     = tileDesc_h->loGC();
        const IntVect       hiGC_h     = tileDesc_h->hiGC();
        IntVect             hiF;
        Real*               dUnk_h     = tileDesc_h->dataPtr();
        assert(dUnk_h);

        // get item shape
        const DataShape& shape = getItemShape(n);

        // pack DELTAS
        siz = shape.sizeVariable(Variable::DELTAS);
        RealVect* dDeltas_src = static_cast<RealVect*>(getInPointerToPartItemVar_src(MemoryPartition::IN, n, Variable::DELTAS));
        RealVect* dDeltas_trg = static_cast<RealVect*>(getPointerToPartItemVar_trg(MemoryPartition::IN, n, Variable::DELTAS));
        ctx->deltas_d = dDeltas_trg;
        std::memcpy(dDeltas_src, &deltas_h, siz);

        // pack LO
        siz = shape.sizeVariable(Variable::LO);
        IntVect*  dLo_src = static_cast<IntVect*>(getInPointerToPartItemVar_src(MemoryPartition::IN, n, Variable::LO));
        IntVect*  dLo_trg = static_cast<IntVect*>(getPointerToPartItemVar_trg(MemoryPartition::IN, n, Variable::LO));
        ctx->lo_d = dLo_trg;
        std::memcpy(dLo_src, &lo_h, siz);

        // pack HI
        siz = shape.sizeVariable(Variable::HI);
        IntVect*  dHi_src = static_cast<IntVect*>(getInPointerToPartItemVar_src(MemoryPartition::IN, n, Variable::HI));
        IntVect*  dHi_trg = static_cast<IntVect*>(getPointerToPartItemVar_trg(MemoryPartition::IN, n, Variable::HI));
        ctx->hi_d = dHi_trg;
        std::memcpy(dHi_src, &hi_h, siz);

        // pack UNK
        nComp = shape.at(Variable::UNK, SHAPE_DIM-1);
        siz   = shape.sizeVariable(Variable::UNK);
        Real*     dUnk_src = static_cast<Real*>(getInPointerToPartItemVar_src(MemoryPartition::INOUT, n, Variable::UNK));
        Real*     dUnk_trg = static_cast<Real*>(getPointerToPartItemVar_trg(MemoryPartition::INOUT, n, Variable::UNK));
        FArray4D  Unk_trg{dUnk_trg, loGC_h, hiGC_h, nComp};
        std::memcpy(&ctx->UNK_d, &Unk_trg, sizeof(FArray4D));
        std::memcpy(dUnk_src, dUnk_h, siz);

        // pack AUX
        nComp = shape.at(Variable::AUX, SHAPE_DIM-1);
        Real*     dAux_trg = static_cast<Real*>(getPointerToPartItemVar_trg(MemoryPartition::SCRATCH, n, Variable::AUX));
        FArray4D  Aux_trg{dAux_trg, loGC_h, hiGC_h, nComp};
        std::memcpy(&ctx->AUX_d, &Aux_trg, sizeof(FArray4D));

        // pack FCX
        nComp = shape.at(Variable::FCX, SHAPE_DIM-1);
        hiF   = IntVect{LIST_NDIM(hi_h.I()+1, hi_h.J(), hi_h.K())};
        Real*     dFcx_trg = static_cast<Real*>(getPointerToPartItemVar_trg(MemoryPartition::SCRATCH, n, Variable::FCX));
        FArray4D  Fcx_trg{dFcx_trg, lo_h, hiF, nComp};
        std::memcpy(&ctx->FCX_d, &Fcx_trg, sizeof(FArray4D));

#if NDIM >= 2
        // pack FCY
        nComp = shape.at(Variable::FCY, SHAPE_DIM-1);
        hiF   = IntVect{LIST_NDIM(hi_h.I(), hi_h.J()+1, hi_h.K())};
        Real*     dFcy_trg = static_cast<Real*>(getPointerToPartItemVar_trg(MemoryPartition::SCRATCH, n, Variable::FCY));
        FArray4D  Fcy_trg{dFcy_trg, lo_h, hiF, nComp};
        std::memcpy(&ctx->FCY_d, &Fcy_trg, sizeof(FArray4D));
#endif
#if NDIM == 3
        // pack FCZ
        nComp = shape.at(Variable::FCZ, SHAPE_DIM-1);
        hiF   = IntVect{LIST_NDIM(hi_h.I(), hi_h.J(), hi_h.K()+1)};
        Real*     dFcz_trg = static_cast<Real*>(getPointerToPartItemVar_trg(MemoryPartition::SCRATCH, n, Variable::FCZ));
        FArray4D  Fcz_trg{dFcz_trg, lo_h, hiF, nComp};
        std::memcpy(&ctx->FCZ_d, &Fcz_trg, sizeof(FArray4D));
#endif
    }

    // finalize packing
    pack_finalize(NDIM - 1 /* nExtraStreams */);
}

void  DataPacket_Hydro_gpu_4::unpack(void) {
    std::size_t   siz;

    // initialize unpacking
    unpack_initialize();

    for (unsigned int n=0; n<items_.size(); ++n) {
        // get tile information (host)
        Tile*               tileDesc_h = items_[n].get();
        assert(tileDesc_h);
        Real*               dUnk_h     = tileDesc_h->dataPtr();
        assert(dUnk_h);

        // get item shape
        const DataShape& shape = getItemShape(n);

        // unpack UNK
        siz = shape.sizeVariable(Variable::UNK);
        Real* dUnk_src = static_cast<Real*>(getOutPointerToPartItemVar_src(MemoryPartition::INOUT, n, Variable::UNK));
        std::memcpy(dUnk_h, dUnk_src, siz);
    }

    // finalize unpacking
    unpack_finalize();
}

void  DataPacket_Hydro_gpu_4::setupItemShapes(void) {
    assert(0 < items_.size());

    // create shape
    std::size_t  shapeEntries[Variable::_N * SHAPE_DIM];
    std::memcpy(shapeEntries, VARIABLE_SHAPES, sizeof(std::size_t) * Variable::_N * SHAPE_DIM);
    shapeEntries[Variable::ICTX*SHAPE_DIM    ] = sizeof(ItemContext);
    shapeEntries[Variable::ICTX*SHAPE_DIM + 1] = items_.size();
    //TODO ^ this does not look like the best solution

    // set uniform shape for all tiles
    itemShapes_.emplace_back(Variable::_N, SHAPE_DIM);
    itemShapes_.back().set(shapeEntries);
    //itemShapes_.back().setPadding(...); //TODO add padding for memalign
    itemShapes_.back().setPartition(VARIABLE_PARTITIONS);
}

void  DataPacket_Hydro_gpu_4::clearItemShapes(void) {
    assert(1 == itemShapes_.size());

    // destroy list of shapes (destructor for shapes is called implicitly)
    itemShapes_.clear();
}

}

