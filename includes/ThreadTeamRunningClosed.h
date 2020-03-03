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

template<typename W, class T>
class ThreadTeamRunningClosed : public ThreadTeamState<W,T> {
public:
    ThreadTeamRunningClosed(T* team);
    ~ThreadTeamRunningClosed(void)                { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::RUNNING_CLOSED_QUEUE;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startTask_NotThreadsafe(
                            TASK_FCN<W> fcn,
                            const unsigned int nThreads,
                            const std::string& teamName, 
                            const std::string& taskName) override;
    std::string     enqueue_NotThreadsafe(const W& work) override;
    std::string     closeTask_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override;

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

