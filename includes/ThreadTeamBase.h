/**
 * \file    ThreadTeamBase.h
 *
 * \brief An abstract base class for the ThreadTeam class that specifies the
 * interface for a Thread Subscriber.
 *
 * A Thread Publisher should be capable of communicating the transition to Idle
 * of one its threads to any other ThreadTeam.  In particular, the data type of
 * the subscriber and publisher need not be the same.  This base class is
 * defined without specifying a data type and therefore allows for the necessary
 * polymorphic use of ThreadTeam objects with regard to thread publishing.
 * 
 */

#ifndef THREAD_TEAM_BASE_H__
#define THREAD_TEAM_BASE_H__

class ThreadTeamBase {
public:
    virtual void increaseThreadCount(const unsigned int nThreads) = 0;

protected:
    ThreadTeamBase(void) {}
    virtual ~ThreadTeamBase(void) {}

private:
    ThreadTeamBase(ThreadTeamBase&) = delete;
    ThreadTeamBase(const ThreadTeamBase&) = delete;
    ThreadTeamBase(ThreadTeamBase&&) = delete;
    ThreadTeamBase& operator=(ThreadTeamBase&) = delete;
    ThreadTeamBase& operator=(const ThreadTeamBase&) = delete;
    ThreadTeamBase& operator=(ThreadTeamBase&&) = delete;
};

#endif

