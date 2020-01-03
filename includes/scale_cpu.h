#ifndef SCALE_CPU_H__
#define SCALE_CPU_H__

#include <string>

#include "Block.h"

namespace ThreadRoutines {
    void scale_cpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif

