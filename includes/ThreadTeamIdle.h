/**
 * \file    ThreadTeamIdle.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Idle mode.
 */

#ifndef THREAD_TEAM_IDLE_H__
#define THREAD_TEAM_IDLE_H__

#include "ThreadTeam.h"
#include "ThreadTeamState.h"

class ThreadTeamIdle : public ThreadTeamState {
public:
    ThreadTeamIdle(ThreadTeam* team);
    ~ThreadTeamIdle(void);

    ThreadTeam::teamMode  mode(void) const;

    void                  increaseThreadCount(const unsigned int nThreads);
    std::string           startTask_NotThreadsafe(TASK_FCN* fcn,
                                    const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName);
    std::string           enqueue_NotThreadsafe(const int work);
    void                  closeTask(void);
    std::string           wait_NotThreadsafe(void);

protected:
    std::string  isStateValid_NotThreadSafe(void) const;

private:
    // Disallow copying of objects to create new objects
    ThreadTeamIdle& operator=(const ThreadTeamIdle& rhs);
    ThreadTeamIdle(const ThreadTeamIdle& other);

    ThreadTeam*    team_;
};

#endif

