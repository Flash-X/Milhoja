#ifndef ERROR_EST_MULTIPLE_H__
#define ERROR_EST_MULTIPLE_H__

#include <memory>

#include <Milhoja_Tile.h>

namespace Simulation {
    void errorEstMultiple(std::shared_ptr<milhoja::Tile> tileDesc, int* tptr);
}

#endif

