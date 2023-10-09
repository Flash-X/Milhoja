#ifndef SIMULATION_H__
#define SIMULATION_H__

#include <string>
#include <vector>

#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Tile.h>

namespace Simulation {
    void setInitialConditions_noRuntime(milhoja::Tile* tileDesc);
}

namespace sim {
    std::vector<std::string>   getVariableNames(void);

    void setInitialConditions(const milhoja::IntVect& lo,
                              const milhoja::IntVect& hi,
                              const unsigned int level,
                              const milhoja::FArray1D& xCoords,
                              const milhoja::FArray1D& yCoords,
                              const milhoja::FArray1D& zCoords,
                              const milhoja::RealVect& deltas,
                              milhoja::FArray4D& solnData);
    void setInitialConditions_topHat(const milhoja::IntVect& lo,
                                     const milhoja::IntVect& hi,
                                     const unsigned int level,
                                     const milhoja::FArray1D& xCoords,
                                     const milhoja::FArray1D& yCoords,
                                     const milhoja::FArray1D& zCoords,
                                     const milhoja::RealVect& deltas,
                                     milhoja::FArray4D& solnData);
}

#endif

