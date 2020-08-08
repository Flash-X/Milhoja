#ifndef ERROR_EST_BLANK_H__
#define ERROR_EST_BLANK_H__

#include "Grid.h"
#include "Tile.h"

#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

namespace Simulation {
    void errorEstBlank(int lev, amrex::TagBoxArray& tags, Real time,
                       int ngrow, std::shared_ptr<Tile> tileDesc);
}

#endif

