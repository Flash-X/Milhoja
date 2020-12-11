#ifndef EOS_H__
#define EOS_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "FArray4D.h"

#include "Flash.h"

namespace Eos {
    // lo/hi can be any two corners, including loGC/hiGC
    void idealGammaDensIe(const orchestration::IntVect& lo,
                          const orchestration::IntVect& hi,
                          orchestration::FArray4D& solnData);
};

#endif

