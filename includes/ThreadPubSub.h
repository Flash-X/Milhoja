/**
 * \file    ThreadPubSub.h
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

#ifndef THREAD_PUB_SUB_H__
#define THREAD_PUB_SUB_H__

#include <string>

namespace orchestration {

class ThreadPubSub {
public:
    virtual void increaseThreadCount(const unsigned int nThreads) = 0;

    virtual std::string   attachThreadReceiver(ThreadPubSub* receiver);
    virtual std::string   detachThreadReceiver(void);

protected:
    ThreadPubSub(void);
    virtual ~ThreadPubSub(void);

    ThreadPubSub*   threadReceiver_; //!< Thread team to notify when threads terminate

private:
    ThreadPubSub(ThreadPubSub&) = delete;
    ThreadPubSub(const ThreadPubSub&) = delete;
    ThreadPubSub(ThreadPubSub&&) = delete;
    ThreadPubSub& operator=(ThreadPubSub&) = delete;
    ThreadPubSub& operator=(const ThreadPubSub&) = delete;
    ThreadPubSub& operator=(ThreadPubSub&&) = delete;
};

}

#endif

