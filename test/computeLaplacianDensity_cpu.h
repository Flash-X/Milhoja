#ifndef COMPUTE_LAPLACIAN_DENSITY_CPU_H__
#define COMPUTE_LAPLACIAN_DENSITY_CPU_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void computeLaplacianDensity_cpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif

