/**
 * \file    ThreadTeamRunningClosed.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Running & Queue Closed mode.
 */

#ifndef THREAD_TEAM_RUNNING_CLOSED_H__
#define THREAD_TEAM_RUNNING_CLOSED_H__

#include "ThreadTeam.h"
#include "ThreadTeamState.h"

class ThreadTeamRunningClosed : public ThreadTeamState {
public:
    ThreadTeamRunningClosed(ThreadTeam* team);
    ~ThreadTeamRunningClosed(void);

    ThreadTeam::teamMode  mode(void) const;

    void                  increaseThreadCount(const unsigned int nThreads);
    void                  startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName);
    void                  enqueue(const int work);
    void                  closeTask(void);
    void                  wait(void);

    void                  attachThreadReceiver(ThreadTeam* receiver);
    void                  detachThreadReceiver(void);

    void                  attachWorkReceiver(ThreadTeam* receiver);
    void                  detachWorkReceiver(void);

private:
    // Disallow copying of objects to create new objects
    ThreadTeamRunningClosed& operator=(const ThreadTeamRunningClosed& rhs);
    ThreadTeamRunningClosed(const ThreadTeamRunningClosed& other);

    ThreadTeam*    team_;
};

#endif

