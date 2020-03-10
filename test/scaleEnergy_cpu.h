#ifndef SCALE_ENERGY_CPU_H__
#define SCALE_ENERGY_CPU_H__

#include <string>

#include "Tile.h"

namespace ThreadRoutines {
    void scaleEnergy_cpu(const int tId, Tile& tileDesc);
}

#endif

