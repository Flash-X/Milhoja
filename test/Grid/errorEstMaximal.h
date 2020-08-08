#ifndef ERROR_EST_MAXIMAL_H__
#define ERROR_EST_MAXIMAL_H__

#include "Grid.h"
#include "Tile.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

namespace Simulation {
    void errorEstMaximal(int lev, amrex::TagBoxArray& tags, Real time,
                     int ngrow, std::shared_ptr<Tile> tileDesc);
}

#endif

