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
 * Please refer to the Orchestration System design documents for more
 * information regarding the general thread team design.
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
 * Waiting, and Computing states.
 *
 * Each method in this class whose behavior is implemented by calling the
 * associated method of the state_ member has a similar structure.
 *   - Acquire the team's mutex
 *   - Do all error checking that should be done for this call regardless of the
 *     team's current state
 *   - Call the state_ member's non-threadsafe version of the method
 *   - Error check this and release the mutex
 * It is important that these routines (and not the state_ methods) acquire the
 * routines.  If we were to call the state_ method and then ask for the mutex,
 * then it is possible that we call the method associate with one mode, but get
 * the mutex after the mode has transitioned.
 *
 * The implementations of the work/thread publisher/subscriber design aspects
 * are one-directional versions of the Observer design pattern (Pp. 293).
 *
 * \warning Client must be certain that subscriber/publisher chains are not
 * setup that code result in deadlocks or infinite loops.
 *
 */

#ifndef THREAD_TEAM_H__
#define THREAD_TEAM_H__

#include <queue>
#include <string>
#include <fstream>

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
    // Note that Terminating is not a mode in the EFSM design.  However,
    // we implement it as if it were to handle the termination of the EFSM as
    // part of the State design pattern.
    enum teamMode   {MODE_IDLE,
                     MODE_TERMINATING,
                     MODE_RUNNING_OPEN_QUEUE,
                     MODE_RUNNING_CLOSED_QUEUE,
                     MODE_RUNNING_NO_MORE_WORK};

    ThreadTeam(const unsigned int nMaxThreads,
               const unsigned int id,
               const std::string& logFilename);
    virtual ~ThreadTeam(void);

    // State-independent methods
    unsigned int             nMaximumThreads(void) const;
    ThreadTeam::teamMode     mode(void);
    void                     stateCounts(unsigned int* N_idle,
                                         unsigned int* N_wait,
                                         unsigned int* N_comp,
                                         unsigned int* N_work);

    // State-dependent methods whose behavior is implemented by objects
    // derived from ThreadTeamState
    void         increaseThreadCount(const unsigned int nThreads);
    void         startTask(TASK_FCN* fcn,
                           const unsigned int nThreads,
                           const std::string& teamName, 
                           const std::string& taskName);
    void         enqueue(const int work);
    void         closeTask(void);
    void         wait(void);

    // State-dependent methods whose simple dependence is handled by this class
    void         attachThreadReceiver(ThreadTeam* receiver);
    void         detachThreadReceiver(void);

    void         attachWorkReceiver(ThreadTeam* receiver);
    void         detachWorkReceiver(void);

protected:
    // Structure used to pass data to each thread at thread creation
    struct ThreadData {
        unsigned int   tId = -1;
        ThreadTeam*    team = nullptr;
    };

    // Routine executed by each thread in the team
    // TODO: Can this be split out as a seperate function in its own file?
    //       This would help with separation of concerns and to keep the file
    //       size down.
    static void* threadRoutine(void*);

    std::string  getModeName(const teamMode mode) const;

    // Code that calls these should acquire teamMutex_ before the call
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

    std::string       logFilename_;
#ifdef VERBOSE
    std::ofstream     logFile_; 
#endif
};

#endif

