#ifndef ERROR_EST_BLANK_H__
#define ERROR_EST_BLANK_H__

#include <memory>

#include <Milhoja_Tile.h>

namespace Simulation {
    void errorEstBlank(std::shared_ptr<milhoja::Tile> tileDesc, int* tptr);
}

#endif

