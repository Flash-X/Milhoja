#ifndef ERROR_EST_BLANK_H__
#define ERROR_EST_BLANK_H__

#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    void errorEstBlank(std::shared_ptr<Tile> tileDesc, int* tptr);
}

#endif

