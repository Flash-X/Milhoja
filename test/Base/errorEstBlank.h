#ifndef ERROR_EST_BLANK_H__
#define ERROR_EST_BLANK_H__

#include "Tile.h"
#include "Grid_REAL.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    Real errorEstBlank(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter);
}

#endif

