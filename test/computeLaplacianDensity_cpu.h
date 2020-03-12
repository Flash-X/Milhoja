#ifndef COMPUTE_LAPLACIAN_DENSITY_CPU_H__
#define COMPUTE_LAPLACIAN_DENSITY_CPU_H__

#include <string>

#include "Tile.h"

namespace ThreadRoutines {
    void computeLaplacianDensity_cpu(const int tId, Tile* tileDesc);
}

#endif

