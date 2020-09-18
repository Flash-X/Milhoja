#ifndef SCALE_ENERGY_BLOCK_H__
#define SCALE_ENERGY_BLOCK_H__

#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"

namespace ThreadRoutines {
    void scaleEnergy_block(const orchestration::IntVect& lo,
                           const orchestration::IntVect& hi,
                           const orchestration::FArray1D& xCoords,
                           const orchestration::FArray1D& yCoords,
                           orchestration::FArray4D& f);
}

#endif

