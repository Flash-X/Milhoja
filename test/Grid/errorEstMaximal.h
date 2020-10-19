#ifndef ERROR_EST_MAXIMAL_H__
#define ERROR_EST_MAXIMAL_H__

#include "Grid_REAL.h"
#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    Real errorEstMaximal(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter);
}

#endif

