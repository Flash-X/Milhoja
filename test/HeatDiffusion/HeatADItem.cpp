#include "HeatADItem.h"
#include "HeatAD.h"

#include <cmath>
#include <algorithm>

#include "Tile.h"
#include "Driver.h"
#include "Flash.h"

void HeatADItem::advanceSolution_tile_cpu(const int tId,
                                          orchestration::DataItem* dataItem) {

    using namespace orchestration;

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    const IntVect       lo       = tileDesc->lo();
    const IntVect       hi       = tileDesc->hi();
    FArray4D            solnData = tileDesc->data();
    RealVect            deltas   = tileDesc->deltas();

    HeatAD::diffusion(solnData,deltas,HeatAD::alpha,lo,hi);
    HeatAD::solve(solnData,Driver::dt,lo,hi);

}

