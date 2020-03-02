/**
 * \file    runtimeTask.h
 *
 * Define the interface that must be used for all functions that will be given
 * to a thread team as a task.
 *
 * Note that the only central parameter is the unit of work on which to apply
 * the task for each invocation of the function.
 */

#ifndef RUNTIME_TASK_H__
#define RUNTIME_TASK_H__

#include <string>

template<typename W>
using TASK_FCN = void (*)(const unsigned int tId,
                          const std::string& name,
                          const W& work);

#endif

