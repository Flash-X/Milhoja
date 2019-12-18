#ifndef GPU_THREAD_ROUTINE_H__
#define GPU_THREAD_ROUTINE_H__

#include <string>

namespace ThreadRoutines {
    void gpu(const unsigned int tId, const std::string& name, const int work);
}

#endif

