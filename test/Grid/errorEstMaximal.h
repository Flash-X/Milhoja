#ifndef ERROR_EST_MAXIMAL_H__
#define ERROR_EST_MAXIMAL_H__

#include "Tile.h"

#include <memory>

using namespace orchestration;

namespace Simulation {
    void errorEstMaximal(std::shared_ptr<Tile> tileDesc, int* tptr);
}

#endif

