/**
 * \file    ThreadTeam.h
 *
 * \brief A class that implements a team of threads for orchestrating the
 * execution on a heterogeneous node of a single task across a given set of
 * units of tiles.
 *
 * The class implements a team of threads that is specifically designed for use
 * by the Orchestration Runtime, which schedules a single task with the thread
 * team for execution in a single cycle.  All public methods are thread-safe.
 *
 * The team is designed and implemented as an extended finite state machine
 * (EFSM).  The mode (qualitative) portion of the EFSM's state must be in one
 * and only one of
 *   - Idle,
 *   - Running with the work queue open
 *     (i.e. able to accept more units of work),
 *   - Running with the work queue closed
 *     (i.e. no more units of work can be added),
 *   - Running with no more pending work
 *     (i.e. there are no more units of work in the queue, but at least one
 *     thread is still applying the task to a unit of work), or
 *   - Terminating
 * at any point in time.  The internal state variables (quantitative) for the
 * EFSM are the number of
 *   - pending units of work in the team's queue,
 *   - Idle threads,
 *   - Waiting threads,
 *   - Computing threads, and
 *   - Terminating threads.
 * The events that trigger state transitions are
 *   - an external call to startTask()
 *     (Idle -> Running & Open),
 *   - an external call to closeTask()
 *     (Running & Open -> Running Closed),
 *   - a thread determines that the queue is empty
 *     (Running & Closed -> Running & No Pending Work),
 *   - all threads have transitioned to Idle
 *     (Running & No Pending Work -> Idle), and
 *   - an external call results in the destruction of the thread team object
 *     (Idle -> Terminating).
 * At construction, the thread team starts the maximum number of threads
 * allotted to it and these persist until the team is destroyed.  The EFSM
 * starts in the Idle mode with all threads in the Idle state and an empty
 * queue.
 *
 * External code starts an execution cycle by calling startTask and at the same
 * time indicates what task is to be executed by the team as well as how many
 * threads in the team shall be activated immediately to start applying the
 * task.  After startTask has been called, external code can use the enqueue
 * method to give to the team tiles on which to apply its task.  These tiles are
 * passed to enqueue one unit of work at a time where a unit of work could be a
 * single tile (e.g. if the task will have the team run computations on a CPU)
 * or a data packet containing multiple tiles (e.g. if the task will have the
 * team run computations on an accelerator).  The external code indicates to the
 * EFSM that all tiles on which to operate during this cycle have already been
 * enqueued.  The EFSM then continues its application of the task to all
 * remaining given tiles and automatically transitions back to Idle once work
 * and, therefore, the execution cycle finishes.  An external thread can be
 * blocked until the task execution cycle finishes by calling the team's wait
 * method.
 *
 * There is no restriction that a single thread team must execute the same task
 * with every execution cycle.  Similarly, there is no restriction that a team
 * must be dedicated to running computation-heavy code on only a single piece of
 * hardware (e.g. only on the CPU or only on the GPU).  Rather, the thread team
 * only knows to execute the code behind a function pointer, which can include
 * kernel launches on an accelerator.  It is possible for that code to execute
 * computationally-heavy code on a mixture of available HW on the node.
 * However, our present best guess is that restricting each task to run
 * principally on a single device is likely clean, simple, elegant, and
 * maintainable.
 *
 * The EFSM is implemented using the State design pattern presented in the Gang
 * of Four's famous Design Patterns book (Pp. 305).  This class exposes the
 * interface of the EFSM to client code and the client code should only
 * instantiate this class.  Each of the ThreadTeam* classes derived from
 * ThreadTeamState are instantiated once by this class and are used to provide
 * state-specific behavior of the team.  This design was chosen as the response
 * to events, outputs, and error checking of the EFSM are logically partitioned
 * across the modes of the EFSM and are therefore largely encapsulated within
 * each class.  This class
 *   - declares and houses all data members needed to implement the EFSM,
 *   - initializes the EFSM in the correct Idle state,
 *   - handles the destruction of the EFSM, and
 *   - defines how all threads in the team shall behave.
 * Note that each thread in the team is a FSM that moves between the Idle,
 * Waiting, Computing, and Terminating states.  The hope is that studying,
 * analysing, and maintaining the EFSM will hopefully be easier thanks to the
 * use of this design.  However, the runtime efficiency of this design has not
 * yet been studied.
 *
 * It is envisioned that the runtime may instantiate multiple thread teams with
 * each team executing a potentially different task.  See the OrchestrationRuntime
 * documentation for more information regarding this intent.
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
 * subscriber team, the publisher automatically calls the subscriber's closeTask
 * method.
 *
 * Note that a thread publisher can only have one thread subscriber and a work
 * publisher can only have one work subscriber.  However, a thread team can be
 * both a thread subscriber and a thread publisher.  Also, a thread team can be
 * a thread subscriber for one team and a work subscriber for another.
 *
 * The implementations of the publisher/subscriber design aspects are
 * one-directional versions of the Observer design pattern (Pp. 293).
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

#include "runtimeTask.h"

class ThreadTeamState;
class ThreadTeamIdle;
class ThreadTeamTerminating;
class ThreadTeamRunningOpen;
class ThreadTeamRunningClosed;
class ThreadTeamRunningNoMoreWork;

class ThreadTeam {
public:
    //***** Extended Finite State Machine State Definition
    // Qualitative state Modes
    enum teamMode   {MODE_IDLE,
                     MODE_TERMINATING,
                     MODE_RUNNING_OPEN_QUEUE,
                     MODE_RUNNING_CLOSED_QUEUE,
                     MODE_RUNNING_NO_MORE_WORK};

    ThreadTeam(const unsigned int nMaxThreads, const unsigned int id);
    virtual ~ThreadTeam(void);

    // State-independent methods
    unsigned int             nMaximumThreads(void) const;
    ThreadTeam::teamMode     mode(void) const;

    // State-dependent methods whose behavior is implemented by objects
    // derived from ThreadTeamState
    void         increaseThreadCount(const unsigned int nThreads);
    void         startTask(TASK_FCN* fcn, const unsigned int nThreads,
                           const std::string& teamName, 
                           const std::string& taskName);
    void         enqueue(const int work);
    void         closeTask(void);
    void         wait(void);

    void         attachThreadReceiver(ThreadTeam* receiver);
    void         detachThreadReceiver(void);

    void         attachWorkReceiver(ThreadTeam* receiver);
    void         detachWorkReceiver(void);

protected:
    // Structure used to pass data to each thread 
    struct ThreadData {
        unsigned int   tId = -1;       //!< ID that each thread uses for logging
        ThreadTeam*    team = nullptr; //!< Pointer to team object that starts/owns the thread
    };

    // Routine that each thread in the team runs
    static void* threadRoutine(void*);

    std::string  getModeName(const teamMode mode) const;
    std::string  setMode_NotThreadsafe(const teamMode nextNode);
    std::string  printState_NotThreadsafe(const std::string& method,
                                          const unsigned int tId,
                                          const std::string& msg) const;

private:
    // Disallow copying of objects to create new objects
    ThreadTeam& operator=(const ThreadTeam& rhs);
    ThreadTeam(const ThreadTeam& other);

    // State Design Pattern - ThreadTeamState derived classes need direct access
    //                        to protected methods and private data members
    friend class ThreadTeamIdle;
    friend class ThreadTeamTerminating;
    friend class ThreadTeamRunningOpen;
    friend class ThreadTeamRunningClosed;
    friend class ThreadTeamRunningNoMoreWork;

    //***** Extended Finite State Machine State Definition
    // Qualitative State Mode
    // Encoded in ThreadTeamState instance pointed to by state_
    ThreadTeamState*              state_;
    ThreadTeamIdle*               stateIdle_;
    ThreadTeamTerminating*        stateTerminating_;
    ThreadTeamRunningOpen*        stateRunOpen_;
    ThreadTeamRunningClosed*      stateRunClosed_;
    ThreadTeamRunningNoMoreWork*  stateRunNoMoreWork_;

    // Quantitative Internal State Variables
    unsigned int   N_idle_;      /*!< Number of threads that are
                                  *   actually Idling
                                  *   (i.e. waiting on activateThread_)) */
    unsigned int   N_wait_;      /*!< No work at last transition.
                                  *   Waiting for transitionThread_. */
    unsigned int   N_comp_;      /*!< Found work at last transition and
                                  *   currently applying task to work.
                                  *   It will transition
                                  *   (computingFinished event) when it
                                  *   finishes that work. */
    unsigned int   N_terminate_; /*!< Number of threads that are terminating
                                  *   or that have terminated. */

    std::queue<int>   queue_;   //!< Internal queue of pending work.

    //***** Data members not directly related to state
    // Client code could set the number of Idle threads by calling startTask()
    // and increaseThreadCount().  However, there is a non-zero delay for the
    // threads to transition to active.
    //
    // If client code calls increaseThreadCount(), we need to throw an error if
    // the increased count would exceed the number of Idle threads in the team.
    // Therefore, we cannot rely on the number of actual Idle threads to do this
    // error checking.  This tracks the number of Idle threads that will be
    // activated but that have not yet received the signal to activate.
    unsigned int      N_to_activate_;

    unsigned int      nMaxThreads_;        //!< Number of threads in team won't exceed this
    unsigned int      id_;                 //!< (Hopefully) unique ID for team
    std::string       hdr_;                //!< Short name of team for logging
    std::string       taskName_;           //!< Short name of task for logging

    pthread_attr_t    attr_;               //!< All threads setup with this attribute
    pthread_mutex_t   teamMutex_;          //!< Use to access members
    pthread_cond_t    threadStarted_;      //!< Each thread emits this signal upon starting
    pthread_cond_t    activateThread_;     //!< Ask idling threads to transition state
    pthread_cond_t    transitionThread_;   //!< Ask waiting threads to transition state
    pthread_cond_t    threadTerminated_;   //!< Each thread emits this signal upon termination
    pthread_cond_t    unblockWaitThread_;  //!< Wake single thread blocked by calling wait()

    TASK_FCN*         taskFcn_;            /*!< Computational task to be applied to
                                            *   all units of enqueued work */

    ThreadTeam*       threadReceiver_;     //!< Thread team to notify when threads terminate
    ThreadTeam*       workReceiver_;       //!< Thread team to pass work to when finished

    // Keep track of when wait() is blocking and when it is released
    bool              isWaitBlocking_;     //!< Only a single thread can be blocked 
};

#endif

