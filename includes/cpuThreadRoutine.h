#ifndef CPU_THREAD_ROUTINE_H__
#define CPU_THREAD_ROUTINE_H__

#include <iostream>

#include "Block.h"

namespace ThreadRoutines {
    void cpu(const unsigned int tId, const std::string& name, Block& block);
}

#endif

