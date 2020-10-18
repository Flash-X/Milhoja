#ifndef ERROR_EST_ADV_H__
#define ERROR_EST_ADV_H__

#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    Real errorEstAdv(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter);
}

#endif

