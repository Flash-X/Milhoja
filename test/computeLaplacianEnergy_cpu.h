#ifndef COMPUTE_LAPLACIAN_ENERGY_CPU_H__
#define COMPUTE_LAPLACIAN_ENERGY_CPU_H__

#include <string>

#include "Tile.h"

namespace ThreadRoutines {
    void computeLaplacianEnergy_cpu(const int tId, Tile& tileDesc);
}

#endif

