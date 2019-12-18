/**
 * \file    ThreadTeam.h
 *
 * \brief A class that implements a team of threads using pthreads.
 *
 * The class implements a team of threads that is specifically designed for use
 * by the Orchestration Runtime.  In particular, 
 *   - the team is set up at instantiation with an upper bound on the total
 *     number of threads that can exist at any point in time,
 *   - a fixed, given number of threads are activated when the startTask()
 *     method is invoked,
 *   - client code provides the thread team work by pushing individual units of
 *     work with the enqueue() method,
 *   - client code indicates to the team when all work for the current task have
 *     been pushed with enqueue() via the closeTask() method,
 *   - the calling thread waits for work on all enqueued units of work to finish
 *     by calling the blocking method wait(), and
 *   - the wait() method returns after computations have finished for all units
 *     of work added to the queue and all threads have terminated.
 * 
 * All public methods are thread-safe.
 *
 * Client code may enqueue units of work before or after calling the startTask()
 * method.  However, no units of work may be enqueued after the closeTask() method
 * has been called and before the wait() method returns.
 *
 * It is envisioned that the runtime may instantiate multiple thread teams with
 * each team executing a single independent task.
 *
 * To control the number of active threads running in the host at any point in
 * time, a thread team can become a thread subscriber of another thread team so
 * that the subscriber is notified that it can add another thread to its team
 * each time a thread terminates in the publisher thread's team.
 *
 * In addition a thread team can also be setup as a work subscriber of another
 * thread team so that when the publishing thread finishes a unit of work, the
 * result is automatically enqueued on the subscriber team's queue.  When the
 * publisher has finished all its work and enqueued all work units in the
 * subscriber team, the publisher automatically closes the subscriber's task.
 *
 * A thread team can be a thread subscriber for one team and a work subscriber
 * for another.
 *
 * The wait() method for a publisher team must be called before the wait()
 * method of its subscriber team.
 *
 * \warning Client must be certain that subscriber/publisher chains are not
 * setup that code result in deadlocks or infinite loops.
 *
 */

#ifndef THREAD_TEAM_H__
#define THREAD_TEAM_H__

#include <queue>
#include <string>
#include <pthread.h>

#include "runtimeTask.h"

// TODO: Make this class a templated class where the template type is the type
// of the unit of work.
class ThreadTeam {
public:
    ThreadTeam(const unsigned int nMaxThreads, const std::string name);
    ~ThreadTeam(void);

    void         increaseThreadCount(const unsigned int nThreads);

    unsigned int nMaximumThreads(void) const;
    std::string  name(void) const;

    void         startTask(TASK_FCN* fcn, const unsigned int nThreads);
    void         enqueue(const int work);
    void         closeTask(void);
    void         wait(void);

    void         attachThreadReceiver(ThreadTeam* receiver);
    void         detachThreadReceiver(void);

    void         attachWorkReceiver(ThreadTeam* receiver);
    void         detachWorkReceiver(void);

    void         printState(const unsigned int tId) const;

protected:
    static void* threadRoutine(void*);
  
    unsigned int nIdleThreads(void) const;

    // A state of RUNNING_CLOSED_QUEUE and an empty queue imply that the
    // current task has finished.
    enum teamState   {TEAM_IDLE,              //!< Thread team is idle
                      RUNNING_OPEN_QUEUE,     /*!< Thread team executing but work
                                               *   can still be queued */
                      RUNNING_CLOSED_QUEUE};  /*!< Thread team executing and no
                                               *   more work can be added */

    enum threadState {STARTING,     //!< Thread is created & starting
                      THREAD_IDLE,  //!< Thread is not in use
                      COMPUTING,    //!< Thread is executing computations on unit of work
                      WAITING,      /*!< Thread is waiting for a unit of work
                                     *   to become available */
                      TERMINATING}; //!< Thread is terminating execution

     // Structure used to pass data to each thread 
    struct ThreadData {
        unsigned int   tId = -1;       /*!< ID that each thread uses to update its state
                                        *   in threadStates_ */
        ThreadTeam*    team = nullptr; //!< Pointer to team object that starts/owns the thread
    };

private:
    unsigned int      nMaxThreads_;       //!< Number of threads in team won't exceed this
    std::string       name_;              //!< Short name of team for debugging

    bool              isTerminating_;     //!< True if the destructor has been entered

    std::queue<int>   queue_;             //!< Internal queue of work to be done
    teamState         state_;             //!< The current state of the thread team
    threadState*      threadStates_;      //!< The current state of each thread in team

    pthread_attr_t    attr_;              //!< All threads setup with this attribute
    pthread_mutex_t   teamMutex_;         //!< Use to access members
    pthread_cond_t    threadIdling_;      //!< Thread finished work and idling
    pthread_cond_t    activateThread_;    //!< Wake an idle thread for work
    pthread_cond_t    checkQueue_;        /*!< Signal sent to waiting threads to
                                           *   indicate that they should recheck
                                           *   the queue and team state.  This
                                           *   is emitted each time work is enqueued 
                                           *   or if a thread has detected that
                                           *   there is no more work in the
                                           *   queue and that no more work will
                                           *   be added to the queue */
    pthread_cond_t    threadTerminated_;  //!< Each thread emits this signal upon termination

    TASK_FCN*         taskFcn_;           /*!< Computational task to be applied to
                                           *   all units of enqueued work */

    // Make these members so that their values are set in
    // start/increaseThreadCount, but still valid when threads run their routine.
    pthread_t*       threads_;            //!< Array of opaque thread indices
    ThreadData*      threadData_;         //!< Array of arguments passed to thread routine

    ThreadTeam*      threadReceiver_;     //!< Thread team to notify when threads terminate
    ThreadTeam*      workReceiver_;       //!< Thread team to pass work to when finished
};

#endif

