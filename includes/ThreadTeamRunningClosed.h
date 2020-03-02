/**
 * \file    ThreadTeamRunningClosed.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Running & Queue Closed mode.
 */

#ifndef THREAD_TEAM_RUNNING_CLOSED_H__
#define THREAD_TEAM_RUNNING_CLOSED_H__

#include "ThreadTeamState.h"

template<class T>
class ThreadTeamRunningClosed : public ThreadTeamState<T> {
public:
    ThreadTeamRunningClosed(T* team);
    ~ThreadTeamRunningClosed(void)                { };

    ThreadTeamModes::mode  mode(void) const {
        return ThreadTeamModes::RUNNING_CLOSED_QUEUE;
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
    ThreadTeamRunningClosed& operator=(const ThreadTeamRunningClosed& rhs);
    ThreadTeamRunningClosed(const ThreadTeamRunningClosed& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamRunningClosed.cpp"

#endif

