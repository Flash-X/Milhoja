#include "computeLaplacianEnergy.h"

#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_TileWrapper.h>

void ActionRoutines::computeLaplacianEnergy_tile_cpu(const int tId,
                                                     milhoja::DataItem* dataItem) {
    using namespace milhoja;

    TileWrapper*  wrapper = dynamic_cast<TileWrapper*>(dataItem);
    Tile*  tileDesc = wrapper->tile_.get();

    const IntVect   lo      = tileDesc->lo();
    const IntVect   hi      = tileDesc->hi();
    const RealVect  deltas  = tileDesc->deltas();
    FArray4D        U       = tileDesc->data();

    FArray4D        scratch = FArray4D::buildScratchArray4D(lo, hi, 1);

    StaticPhysicsRoutines::computeLaplacianEnergy(lo, hi, U, scratch, deltas);
}

