#ifndef RUNTIME_TASK_H__
#define RUNTIME_TASK_H__

#include <string>

typedef void (TASK_FCN)(const unsigned int tId,
                        const std::string& name,
                        const int work);

#endif

