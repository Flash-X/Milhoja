#ifndef COMPUTE_LAPLACIAN_ENERGY_CPU_H__
#define COMPUTE_LAPLACIAN_ENERGY_CPU_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void computeLaplacianEnergy_cpu(const int tId, Block& block);
}

#endif
