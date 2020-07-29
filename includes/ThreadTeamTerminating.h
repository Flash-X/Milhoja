/**
 * \file    ThreadTeamTerminating.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Terminating mode.
 */

#ifndef THREAD_TEAM_TERMINATING_H__
#define THREAD_TEAM_TERMINATING_H__

#include "ThreadTeamState.h"

namespace orchestration {

class DataItem;
class ThreadTeam;
class ThreadTeamState;

class ThreadTeamTerminating : public ThreadTeamState {
public:
    ThreadTeamTerminating(ThreadTeam* team);
    ~ThreadTeamTerminating(void)                { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::TERMINATING;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startCycle_NotThreadsafe(
                            const RuntimeAction& action,
                            const std::string& teamName) override;
    std::string     enqueue_NotThreadsafe(std::shared_ptr<DataItem>&& dataItem) override;
    std::string     closeQueue_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override {
        return "";
    }

private:
    // Disallow copying/moving
    ThreadTeamTerminating(ThreadTeamTerminating& other)                = delete;
    ThreadTeamTerminating(const ThreadTeamTerminating& other)          = delete;
    ThreadTeamTerminating(ThreadTeamTerminating&& other)               = delete;
    ThreadTeamTerminating& operator=(ThreadTeamTerminating& rhs)       = delete;
    ThreadTeamTerminating& operator=(const ThreadTeamTerminating& rhs) = delete;
    ThreadTeamTerminating& operator=(ThreadTeamTerminating&& rhs)      = delete;

    ThreadTeam*    team_;
};
}

#endif

