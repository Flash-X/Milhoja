#ifndef ERROR_EST_FLASH_H__
#define ERROR_EST_FLASH_H__

#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    Real errorEstFlash(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter);
}

#endif

