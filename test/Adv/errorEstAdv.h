#ifndef ERROR_EST_ADV_H__
#define ERROR_EST_ADV_H__

#include "Grid.h"
#include "Tile.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

namespace Simulation {
    void errorEstAdv(int lev, amrex::TagBoxArray& tags, Real time,
                     int ngrow, std::shared_ptr<Tile> tileDesc);
}

#endif

