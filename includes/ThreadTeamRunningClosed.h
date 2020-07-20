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

namespace orchestration {

template<typename DT, class T>
class ThreadTeamRunningClosed : public ThreadTeamState<DT,T> {
public:
    ThreadTeamRunningClosed(T* team);
    ~ThreadTeamRunningClosed(void)                { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::RUNNING_CLOSED_QUEUE;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startCycle_NotThreadsafe(
                            const RuntimeAction& action,
                            const std::string& teamName) override;
    std::string     enqueue_NotThreadsafe(std::shared_ptr<DT>&& dataItem) override;
    std::string     closeQueue_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override;

private:
    // Disallow copying/moving
    ThreadTeamRunningClosed(ThreadTeamRunningClosed& other)                = delete;
    ThreadTeamRunningClosed(const ThreadTeamRunningClosed& other)          = delete;
    ThreadTeamRunningClosed(ThreadTeamRunningClosed&& other)               = delete;
    ThreadTeamRunningClosed& operator=(ThreadTeamRunningClosed& rhs)       = delete;
    ThreadTeamRunningClosed& operator=(const ThreadTeamRunningClosed& rhs) = delete;
    ThreadTeamRunningClosed& operator=(ThreadTeamRunningClosed&& rhs)      = delete;

    T*    team_;
};
}

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamRunningClosed.cpp"

#endif

