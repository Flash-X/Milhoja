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

// The work must be passed as a pointer for interoperability with Fortran
// interface
template<typename W>
using TASK_FCN = void (*)(const int tId, W* work);

#endif

