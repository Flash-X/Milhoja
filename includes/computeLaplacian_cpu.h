#ifndef COMPUTE_LAPLACIAN_CPU_H__
#define COMPUTE_LAPLACIAN_CPU_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void computeLaplacian_cpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif

