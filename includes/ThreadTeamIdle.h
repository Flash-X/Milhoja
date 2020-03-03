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

template<typename W, class T>
class ThreadTeamIdle : public ThreadTeamState<W, T> {
public:
    ThreadTeamIdle(T* team);
    ~ThreadTeamIdle(void)                 { };

    ThreadTeamMode  mode(void) const override {
        return ThreadTeamMode::IDLE;
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
    ThreadTeamIdle& operator=(const ThreadTeamIdle& rhs);
    ThreadTeamIdle(const ThreadTeamIdle& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamIdle.cpp"

#endif

