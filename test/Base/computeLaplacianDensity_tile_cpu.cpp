#include "computeLaplacianDensity.h"

#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Tile.h>

void ActionRoutines::computeLaplacianDensity_tile_cpu(const int tId,
                                                      milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    const IntVect   lo      = tileDesc->lo();
    const IntVect   hi      = tileDesc->hi();
    const RealVect  deltas  = tileDesc->deltas();
    FArray4D        U       = tileDesc->data();

    FArray4D        scratch = FArray4D::buildScratchArray4D(lo, hi, 1);

    StaticPhysicsRoutines::computeLaplacianDensity(lo, hi, U, scratch, deltas);
}

