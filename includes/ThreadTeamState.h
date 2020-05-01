/**
 * \file    ThreadTeamState.h
 *
 * A pure abstract base class for defining the interface of all ThreadTeam*
 * derived classes that will implement the ThreadTeam state-specific behavior.
 * This design follows the State design pattern.
 *
 * The template variable W defines the unit of work (e.g. a tile, data packet of
 * tiles); T, refers to the main State class in the State design pattern and
 * should therefore always be ThreadTeam<W>.  Note that T is necessary to break
 * a circular dependence with ThreadTeam.h.
 */

#ifndef THREAD_TEAM_STATE_H__
#define THREAD_TEAM_STATE_H__

#include <string>

template<typename W, class T>
class ThreadTeamState {
public:
    ThreadTeamState(void)          {  }
    virtual ~ThreadTeamState(void) {  }

    virtual ThreadTeamMode  mode(void) const = 0;

    virtual std::string     increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads) = 0;
    virtual std::string     startTask_NotThreadsafe(
                                    TASK_FCN fcn,
                                    const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName) = 0;
    virtual std::string     enqueue_NotThreadsafe(W& work, const bool move) = 0;
    virtual std::string     closeTask_NotThreadsafe(void) = 0;

protected:
    friend T;

    virtual std::string     isStateValid_NotThreadSafe(void) const = 0;
};

#endif

