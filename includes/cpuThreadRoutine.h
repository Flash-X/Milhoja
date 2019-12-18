#ifndef CPU_THREAD_ROUTINE_H__
#define CPU_THREAD_ROUTINE_H__

#include <iostream>

namespace ThreadRoutines {
    void cpu(const unsigned int tId, const std::string& name, const int work);
}

#endif

