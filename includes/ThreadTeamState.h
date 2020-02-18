/**
 * \file    ThreadTeamState.h
 *
 * A pure abstract base class for defining the interface of all ThreadTeam*
 * derived classes that will implement the ThreadTeam state-specific behavior.
 * This design follows the State design pattern.
 */

#ifndef THREAD_TEAM_STATE_H__
#define THREAD_TEAM_STATE_H__

#include <string>

#include "ThreadTeam.h"

class ThreadTeamState {
public:
    ThreadTeamState(void)          {  }
    virtual ~ThreadTeamState(void) {  }

    virtual ThreadTeam::teamMode  mode(void) const = 0;

    virtual void                  increaseThreadCount(const unsigned int nThreads) = 0;
    virtual std::string           startTask_NotThreadsafe(TASK_FCN* fcn,
                                            const unsigned int nThreads,
                                            const std::string& teamName, 
                                            const std::string& taskName) = 0;
    virtual std::string           enqueue_NotThreadsafe(const int work) = 0;
    virtual std::string           closeTask_NotThreadsafe(void) = 0;
    virtual std::string           wait_NotThreadsafe(void) = 0;

protected:
    friend class ThreadTeam; 

    virtual std::string           isStateValid_NotThreadSafe(void) const = 0;
};

#endif

