/**
 * \file    ThreadTeamRunningOpen.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Running & Queue Open mode.
 */

#ifndef THREAD_TEAM_RUNNING_OPEN_H__
#define THREAD_TEAM_RUNNING_OPEN_H__

#include "ThreadTeam.h"
#include "ThreadTeamState.h"

class ThreadTeamRunningOpen : public ThreadTeamState {
public:
    ThreadTeamRunningOpen(ThreadTeam* team);
    ~ThreadTeamRunningOpen(void);

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
    ThreadTeamRunningOpen& operator=(const ThreadTeamRunningOpen& rhs);
    ThreadTeamRunningOpen(const ThreadTeamRunningOpen& other);

    ThreadTeam*    team_;
};

#endif

