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

namespace orchestration {

template<typename DT, class T>
class ThreadTeamIdle : public ThreadTeamState<DT, T> {
public:
    ThreadTeamIdle(T* team);
    ~ThreadTeamIdle(void)                 { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::IDLE;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startCycle_NotThreadsafe(
                            const RuntimeAction& action,
                            const std::string& teamName) override;
    std::string     enqueue_NotThreadsafe(DT& dataItem, const bool move) override;
    std::string     closeQueue_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override;

private:
    // Disallow copying/moving
    ThreadTeamIdle(ThreadTeamIdle& other)                = delete;
    ThreadTeamIdle(const ThreadTeamIdle& other)          = delete;
    ThreadTeamIdle(ThreadTeamIdle&& other)               = delete;
    ThreadTeamIdle& operator=(ThreadTeamIdle& rhs)       = delete;
    ThreadTeamIdle& operator=(const ThreadTeamIdle& rhs) = delete;
    ThreadTeamIdle& operator=(ThreadTeamIdle&& rhs)      = delete;

    T*    team_;
};

}
// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamIdle.cpp"

#endif

