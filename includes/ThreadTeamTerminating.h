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

template<class T>
class ThreadTeamTerminating : public ThreadTeamState<T> {
public:
    ThreadTeamTerminating(T* team);
    ~ThreadTeamTerminating(void)                { };

    ThreadTeamModes::mode  mode(void) const {
        return ThreadTeamModes::TERMINATING;
    }

    std::string           increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads);
    std::string           startTask_NotThreadsafe(
                                    TASK_FCN* fcn,
                                    const unsigned int nThreads,
                                    const std::string& teamName, 
                                    const std::string& taskName);
    std::string           enqueue_NotThreadsafe(const int work);
    std::string           closeTask_NotThreadsafe(void);

protected:
    std::string  isStateValid_NotThreadSafe(void) const {
        return "";
    }

private:
    // Disallow copying of objects to create new objects
    ThreadTeamTerminating& operator=(const ThreadTeamTerminating& rhs);
    ThreadTeamTerminating(const ThreadTeamTerminating& other);

    T*    team_;
};

// Include class definition in header since this is a class template
//   => no need to compile the .cpp file directly as part of build
#include "../src/ThreadTeamTerminating.cpp"

#endif

