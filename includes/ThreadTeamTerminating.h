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

template<typename W, class T>
class ThreadTeamTerminating : public ThreadTeamState<W,T> {
public:
    ThreadTeamTerminating(T* team);
    ~ThreadTeamTerminating(void)                { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::TERMINATING;
    }

    std::string     increaseThreadCount_NotThreadsafe(
                            const unsigned int nThreads) override;
    std::string     startTask_NotThreadsafe(
                            const RuntimeAction& action,
                            const std::string& teamName) override;
    std::string     enqueue_NotThreadsafe(W& work, const bool move) override;
    std::string     closeTask_NotThreadsafe(void) override;

protected:
    std::string  isStateValid_NotThreadSafe(void) const override {
        return "";
    }

private:
    // Disallow copying of objects to create new objects
    ThreadTeamTerminating& operator=(const ThreadTeamTerminating& rhs);
    ThreadTeamTerminating(const ThreadTeamTerminating& other);

    T*    team_;
};
}

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamTerminating.cpp"

#endif

