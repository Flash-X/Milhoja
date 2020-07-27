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

namespace orchestration {

class DataItem;
class ThreadTeam;
class ThreadTeamState;

class ThreadTeamRunningOpen : public ThreadTeamState {
public:
    ThreadTeamRunningOpen(ThreadTeam* team);
    ~ThreadTeamRunningOpen(void)                { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::RUNNING_OPEN_QUEUE;
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
    // Disallow copying of objects to create new objects
    ThreadTeamRunningOpen(ThreadTeamRunningOpen& other)                = delete;
    ThreadTeamRunningOpen(const ThreadTeamRunningOpen& other)          = delete;
    ThreadTeamRunningOpen(ThreadTeamRunningOpen&& other)               = delete;
    ThreadTeamRunningOpen& operator=(ThreadTeamRunningOpen& rhs)       = delete;
    ThreadTeamRunningOpen& operator=(const ThreadTeamRunningOpen& rhs) = delete;
    ThreadTeamRunningOpen& operator=(ThreadTeamRunningOpen&& rhs)      = delete;

    ThreadTeam*    team_;
};
}

#endif

