#ifndef SIMULATION_H__
#define SIMULATION_H__

#include <string>
#include <vector>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace Simulation {
    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void setInitialConditions_tile_cpu(const int tId,
                                       orchestration::DataItem* dataItem);
}

namespace sim {
    std::vector<std::string>   getVariableNames(void);

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

