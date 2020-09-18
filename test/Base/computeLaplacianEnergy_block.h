#ifndef COMPUTE_LAPLACIAN_ENERGY_BLOCK_H__
#define COMPUTE_LAPLACIAN_ENERGY_BLOCK_H__

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray4D.h"

#include "constants.h"

namespace ThreadRoutines {
    void computeLaplacianEnergy_block(const orchestration::IntVect* lo_d,
                                      const orchestration::IntVect* hi_d,
                                      orchestration::FArray4D* f_d,
                                      orchestration::FArray4D* scratch_d,
                                      const orchestration::RealVect* deltas_d,
                                      const int streamId);
}

#endif

