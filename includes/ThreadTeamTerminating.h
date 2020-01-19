/**
 * \file    ThreadTeamTerminating.h
 *
 * A concreate ThreadTeamState derived class that implements the correct
 * behavior of the Thead Team extended finite state machine when the machine is
 * in the Terminating mode.
 */

#ifndef THREAD_TEAM_TERMINATING_H__
#define THREAD_TEAM_TERMINATING_H__

#include "ThreadTeam.h"
#include "ThreadTeamState.h"

class ThreadTeamTerminating : public ThreadTeamState {
public:
    ThreadTeamTerminating(ThreadTeam* team);
    ~ThreadTeamTerminating(void);

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
    ThreadTeamTerminating& operator=(const ThreadTeamTerminating& rhs);
    ThreadTeamTerminating(const ThreadTeamTerminating& other);

    ThreadTeam*    team_;
};

#endif
