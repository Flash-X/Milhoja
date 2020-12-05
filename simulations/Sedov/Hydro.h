#ifndef HYDRO_H__
#define HYDRO_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray3D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace Hydro {
    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void advanceSolution_tile_cpu(const int tId,
                                  orchestration::DataItem* dataItem);
};

namespace hy {
    void computeFluxesHll(const orchestration::Real dt,
                          const orchestration::IntVect& lo,
                          const orchestration::IntVect& hi,
                          const orchestration::RealVect& deltas,
                          const orchestration::FArray4D& Uin,
                          orchestration::FArray4D& flX,
                          orchestration::FArray4D& flY,
                          orchestration::FArray4D& flZ,
                          orchestration::FArray3D& auxC);

    void updateSolutionHll(const orchestration::IntVect& lo,
                           const orchestration::IntVect& hi,
                           const orchestration::FArray4D& Uin,
                           orchestration::FArray4D& Uout,
                           const orchestration::FArray4D& flX,
                           const orchestration::FArray4D& flY,
                           const orchestration::FArray4D& flZ);
};

#endif

