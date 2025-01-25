/**
 * \file    Milhoja_ThreadTeamState.h
 *
 * A pure abstract base class for defining the interface of all ThreadTeam*
 * derived classes that will implement the ThreadTeam state-specific behavior.
 * This design follows the State design pattern.
 *
 */

#ifndef MILHOJA_THREAD_TEAM_STATE_H__
#define MILHOJA_THREAD_TEAM_STATE_H__

#include <string>
#include <memory>

#include "Milhoja_ThreadTeamMode.h"
#include "Milhoja_RuntimeAction.h"

namespace milhoja {

class DataItem;
class ThreadTeam;

class ThreadTeamState {
public:
    ThreadTeamState(void)          {  }
    virtual ~ThreadTeamState(void) {  }

    virtual ThreadTeamMode  mode(void) const = 0;

    virtual std::string     increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads) = 0;
    virtual std::string     startCycle_NotThreadsafe(
                                    const RuntimeAction& action,
                                    const std::string& teamName) = 0;
    virtual std::string     enqueue_NotThreadsafe(std::shared_ptr<DataItem>&& dataItem) = 0;
    virtual std::string     closeQueue_NotThreadsafe(void) = 0;

protected:
    friend ThreadTeam;

    virtual std::string     isStateValid_NotThreadSafe(void) const = 0;
};

}

#endif

