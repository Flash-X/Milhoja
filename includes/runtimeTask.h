/**
 * \file    runtimeTask.h
 *
 * Define the interface that must be used for all functions that will be given
 * to a thread team as a task.
 *
 * Note that the only central parameter is the unit of work on which to apply
 * the task for each invocation of the function.
 *
 * \todo If we want to templatize ThreadTeam so that each team object is built
 *       to work with just one unit of work, then we would need more than one
 *       interface and the task definitions would have to be adapted based on
 *       the unit of work.  Can we templatize this interface as well?
 */

#ifndef RUNTIME_TASK_H__
#define RUNTIME_TASK_H__

#include <string>

typedef void (TASK_FCN)(const unsigned int tId,
                        const std::string& name,
                        unsigned int work);

// For the future
//typedef void (TASK_FCN)(const unsigned int tId,
//                        const std::string& name,
//                        const Grid_tile_t& work);

// For the future
//typedef void (TASK_FCN)(const unsigned int tId,
//                        const std::string& name,
//                        const DataPacket& work);

#endif

