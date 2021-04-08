#ifndef HEATAD_H__
#define HEATAD_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray3D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace HeatAD {
    extern orchestration::Real alpha;

    void diffusion(orchestration::FArray4D& solnData,
                   const orchestration::RealVect& deltas,
                   const orchestration::Real diffusion_coeff,
                   const orchestration::IntVect& lo,
                   const orchestration::IntVect& hi);

    void solve(orchestration::FArray4D& solnData,
               const orchestration::Real dt,
               const orchestration::IntVect& lo,
               const orchestration::IntVect& hi);

};

#endif

