#ifndef ERROR_EST_MAXIMAL_H__
#define ERROR_EST_MAXIMAL_H__

#include <memory>

#include <Milhoja_Tile.h>

namespace Simulation {
    void errorEstMaximal(std::shared_ptr<milhoja::Tile> tileDesc, int* tptr);
}

#endif

