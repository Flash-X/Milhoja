#include "computeLaplacianEnergy.h"

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray4D.h"
#include "Tile.h"

void ActionRoutines::computeLaplacianEnergy_tile_cpu(const int tId,
                                                     orchestration::DataItem* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    const IntVect   lo      = tileDesc->lo();
    const IntVect   hi      = tileDesc->hi();
    const RealVect  deltas  = tileDesc->deltas();
    FArray4D        U       = tileDesc->data();

    FArray4D        scratch = FArray4D::buildScratchArray4D(lo, hi, 1);

    StaticPhysicsRoutines::computeLaplacianEnergy(lo, hi, U, scratch, deltas);
}

