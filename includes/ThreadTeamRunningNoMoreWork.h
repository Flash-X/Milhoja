/**
 * \file    ThreadTeamRunningNoMoreWork.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Running but No Pending Work mode.
 */

#ifndef THREAD_TEAM_RUNNING_NO_MORE_WORK_H__
#define THREAD_TEAM_RUNNING_NO_MORE_WORK_H__

#include "ThreadTeamState.h"

template<class T>
class ThreadTeamRunningNoMoreWork : public ThreadTeamState<T> {
public:
    ThreadTeamRunningNoMoreWork(T* team);
    ~ThreadTeamRunningNoMoreWork(void)                { };

    // State-dependent methods
    ThreadTeamModes::mode  mode(void) const {
        return ThreadTeamModes::RUNNING_NO_MORE_WORK;
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
    ThreadTeamRunningNoMoreWork& operator=(const ThreadTeamRunningNoMoreWork& rhs);
    ThreadTeamRunningNoMoreWork(const ThreadTeamRunningNoMoreWork& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamRunningNoMoreWork.cpp"

#endif

