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

namespace orchestration {

class DataItem;
class ThreadTeam;
class ThreadTeamState;

class ThreadTeamRunningNoMoreWork : public ThreadTeamState {
public:
    ThreadTeamRunningNoMoreWork(ThreadTeam* team);
    ~ThreadTeamRunningNoMoreWork(void)                { };

    // State-dependent methods
    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::RUNNING_NO_MORE_WORK;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startCycle_NotThreadsafe(
                            const RuntimeAction& action,
                            const std::string& teamName) override;
    std::string     enqueue_NotThreadsafe(std::shared_ptr<DataItem>&& dataItem) override;
    std::string     closeQueue_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override;

private:
    // Disallow copying/moving
    ThreadTeamRunningNoMoreWork(ThreadTeamRunningNoMoreWork& other)                = delete;
    ThreadTeamRunningNoMoreWork(const ThreadTeamRunningNoMoreWork& other)          = delete;
    ThreadTeamRunningNoMoreWork(ThreadTeamRunningNoMoreWork&& other)               = delete;
    ThreadTeamRunningNoMoreWork& operator=(ThreadTeamRunningNoMoreWork& rhs)       = delete;
    ThreadTeamRunningNoMoreWork& operator=(const ThreadTeamRunningNoMoreWork& rhs) = delete;
    ThreadTeamRunningNoMoreWork& operator=(ThreadTeamRunningNoMoreWork&& rhs)      = delete;

    ThreadTeam*    team_;
};
}

#endif

