#include "computeLaplacianFused.h"

#include <Milhoja_TileWrapper.h>

void ActionRoutines::computeLaplacianFusedKernels_tile_cpu(const int tId,
                                                           milhoja::DataItem* dataItem) {
    using namespace milhoja;

    TileWrapper*  wrapper = dynamic_cast<TileWrapper*>(dataItem);
    Tile*  tileDesc = wrapper->tile_.get();

    const IntVect   lo      = tileDesc->lo();
    const IntVect   hi      = tileDesc->hi();
    const RealVect  deltas  = tileDesc->deltas();
    FArray4D        U       = tileDesc->data();

    FArray4D        scratch = FArray4D::buildScratchArray4D(lo, hi, 2);

    StaticPhysicsRoutines::computeLaplacianFusedKernels(lo, hi, U, scratch, deltas);
}

