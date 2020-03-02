/**
 * \file    ThreadTeamRunningOpen.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Running & Queue Open mode.
 */

#ifndef THREAD_TEAM_RUNNING_OPEN_H__
#define THREAD_TEAM_RUNNING_OPEN_H__

#include "ThreadTeamState.h"

template<typename W, class T>
class ThreadTeamRunningOpen : public ThreadTeamState<W,T> {
public:
    ThreadTeamRunningOpen(T* team);
    ~ThreadTeamRunningOpen(void)                { };

    ThreadTeamModes::mode  mode(void) const {
        return ThreadTeamModes::RUNNING_OPEN_QUEUE;
    }

    std::string           increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads);
    std::string           startTask_NotThreadsafe(
                                    TASK_FCN<W> fcn,
                                    const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName);
    std::string           enqueue_NotThreadsafe(const W& work);
    std::string           closeTask_NotThreadsafe(void);

protected:
    std::string  isStateValid_NotThreadSafe(void) const;

private:
    // Disallow copying of objects to create new objects
    ThreadTeamRunningOpen& operator=(const ThreadTeamRunningOpen& rhs);
    ThreadTeamRunningOpen(const ThreadTeamRunningOpen& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamRunningOpen.cpp"

#endif

