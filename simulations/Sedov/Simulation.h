#ifndef SIMULATION_H__
#define SIMULATION_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace Simulation {
    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void setInitialConditions_tile_cpu(const int tId,
                                       orchestration::DataItem* dataItem);

    //----- FIX RUNTIME PARAMETERS
    constexpr  orchestration::Real    t_0      = 0.0_wp;
    constexpr  orchestration::Real    t_max    = 0.5_wp;
    // For some reason, Visit's metadata server fails for simulation results
    // acquired after 72 steps.
    constexpr  unsigned int           maxSteps = 70;
    // When FLASH-X runs Sedov/2D with a dtInit that is too small, it sets
    // dtInit to one-tenth the CFL-limited dt value for Hydro, which is this
    // value.
    constexpr  orchestration::Real    dtInit   = 5.6922183414086268e-6_wp;
}

namespace sim {
    void setInitialConditions(const orchestration::IntVect& lo,
                              const orchestration::IntVect& hi,
                              const unsigned int level,
                              const orchestration::FArray1D& xCoords,
                              const orchestration::FArray1D& yCoords,
                              const orchestration::FArray1D& zCoords,
                              const orchestration::RealVect& deltas,
                              orchestration::FArray4D& solnData);
    void setInitialConditions_topHat(const orchestration::IntVect& lo,
                                     const orchestration::IntVect& hi,
                                     const unsigned int level,
                                     const orchestration::FArray1D& xCoords,
                                     const orchestration::FArray1D& yCoords,
                                     const orchestration::FArray1D& zCoords,
                                     const orchestration::RealVect& deltas,
                                     orchestration::FArray4D& solnData);
}

#endif

