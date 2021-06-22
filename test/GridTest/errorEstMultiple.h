#ifndef ERROR_EST_MULTIPLE_H__
#define ERROR_EST_MULTIPLE_H__

#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    void errorEstMultiple(std::shared_ptr<Tile> tileDesc, int* tptr);
}

#endif

