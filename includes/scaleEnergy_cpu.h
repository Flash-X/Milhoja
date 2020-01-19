#ifndef SCALE_ENERGY_CPU_H__
#define SCALE_ENERGY_CPU_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void scaleEnergy_cpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif
