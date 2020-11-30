#ifndef ERROR_EST_ADV_H__
#define ERROR_EST_ADV_H__

#include "Tile.h"
#include <memory>

using namespace orchestration;

namespace Simulation {
    void errorEstAdv(std::shared_ptr<Tile> tileDesc, int* tptr);
}

#endif

