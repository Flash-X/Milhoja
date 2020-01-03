#ifndef RUNTIME_TASK_H__
#define RUNTIME_TASK_H__

#include <string>

#include "Block.h"

typedef void (TASK_FCN)(const unsigned int tId,
                        const std::string& name,
                        Block& block);

#endif

