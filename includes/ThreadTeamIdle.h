/**
 * \file    ThreadTeamIdle.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Idle mode.
 */

#ifndef THREAD_TEAM_IDLE_H__
#define THREAD_TEAM_IDLE_H__

#include "ThreadTeamState.h"

template<class T>
class ThreadTeamIdle : public ThreadTeamState<T> {
public:
    ThreadTeamIdle(T* team);
    ~ThreadTeamIdle(void)                 { };

    ThreadTeamModes::mode  mode(void) const {
        return ThreadTeamModes::IDLE;
    }

    std::string           increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads);
    std::string           startTask_NotThreadsafe(
                                    TASK_FCN* fcn,
                                    const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName);
    std::string           enqueue_NotThreadsafe(const int work);
    std::string           closeTask_NotThreadsafe(void);

protected:
    std::string  isStateValid_NotThreadSafe(void) const;

private:
    // Disallow copying of objects to create new objects
    ThreadTeamIdle& operator=(const ThreadTeamIdle& rhs);
    ThreadTeamIdle(const ThreadTeamIdle& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamIdle.cpp"

#endif

