#ifndef COMPUTE_LAPLACIAN_DENSITY_BLOCK_H__
#define COMPUTE_LAPLACIAN_DENSITY_BLOCK_H__

#include "Grid_IntVect.h"
#include "FArray4D.h"

#include "constants.h"

namespace ThreadRoutines {
    void computeLaplacianDensity_block(const orchestration::IntVect* lo_d,
                                       const orchestration::IntVect* hi_d,
                                       orchestration::FArray4D* f_d,
                                       orchestration::FArray4D* scratch_d,
                                       const orchestration::Real deltas_d[MDIM],
                                       const int streamId);
}

#endif

