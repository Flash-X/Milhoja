/**
 * \file    Milhoja_ThreadTeam.h
 *
 * \brief A class that implements a team of threads for orchestrating the
 * execution on a heterogeneous node of a single action across a given set of
 * data items.
 *
 * The class implements a team of threads that is specifically designed for use
 * by the Orchestration Runtime, which schedules a single action with the thread
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
 * Note that at all times each thread in the team is in one and only of the
 * possible thread states Idle, Waiting, or Computing.
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
 * \warning Calling code must be certain that subscriber/publisher chains are not
 * setup that code result in deadlocks or infinite loops.
 *
 */

#ifndef MILHOJA_THREAD_TEAM_H__
#define MILHOJA_THREAD_TEAM_H__

#include <queue>
#include <string>
#include <memory>

#include <pthread.h>

#include "Milhoja_TileWrapper.h"
#include "Milhoja_actionRoutine.h"
#include "Milhoja_RuntimeAction.h"
#include "Milhoja_ThreadTeamMode.h"
#include "Milhoja_RuntimeElement.h"

namespace milhoja {

class DataItem;
class ThreadTeamState;
class ThreadTeamIdle;
class ThreadTeamTerminating;
class ThreadTeamRunningOpen;
class ThreadTeamRunningClosed;
class ThreadTeamRunningNoMoreWork;

class ThreadTeam : public RuntimeElement {
public:
    ThreadTeam(const unsigned int nMaxThreads,
               const unsigned int id);
    virtual ~ThreadTeam(void);

    // State-independent methods
    unsigned int      nMaximumThreads(void) const;
    ThreadTeamMode    mode(void);
    void              stateCounts(unsigned int* N_idle,
                                  unsigned int* N_wait,
                                  unsigned int* N_comp,
                                  unsigned int* N_work);

    // State-dependent methods whose behavior is implemented by objects
    // derived from ThreadTeamState
    void         increaseThreadCount(const unsigned int nThreads) override;
    void         startCycle(const RuntimeAction& action,
                            const std::string& teamName,
                            const bool waitForThreads=false);
    void         enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void         closeQueue(const RuntimeElement* publisher) override;
    void         wait(void);

    // State-dependent methods whose simple dependence is handled by this class
    std::string  attachThreadReceiver(RuntimeElement* receiver) override;
    std::string  detachThreadReceiver(void) override;

    // DEV: Originally this class was templated so that each ThreadTeam
    //      had a definite data type.  In addition, that same template
    //      was used in this interface so that a given Data Publisher
    //      Subscriber pair had to have the same data type.  However, this
    //      forced ThreadTeam's to be too specialized when a Data Subscriber
    //      could be a data splitter, which had an In data type and an Out data
    //      type.
    //      
    //      By created the DataItem ABC class (i.e. swap templates for
    //      polymorphism), ThreadTeams no longer have a definite type and
    //      this interface suffers because it allows for Pub/Sub pairs
    //      with mismatched data type, which is not allowed.  In other words,
    //      the compiler does not protect those who assemble RuntimeElements
    //      into thread team configurations.
    std::string  attachDataReceiver(RuntimeElement* receiver) override;
    std::string  detachDataReceiver(void) override;
    std::string  setReceiverPrototype(const DataItem* prototype) override;

protected:
    constexpr static unsigned int   THREAD_START_STOP_TIMEOUT_SEC = 1;

    // Structure used to pass data to each thread at thread creation
    struct ThreadData {
        unsigned int   tId =  0;
        ThreadTeam*    team = nullptr;
    };

    // Routine executed by each thread in the team
    // This routine is made a class method so that it can have access to
    // protected and private members and methods. 
    static void* threadRoutine(void*);

    std::string  getModeName(const ThreadTeamMode mode) const;

    // Code that calls these should acquire teamMutex_ before the call
    std::string  setMode_NotThreadsafe(const ThreadTeamMode nextNode);
    std::string  printState_NotThreadsafe(const std::string& method,
                                          const unsigned int tId,
                                          const std::string& msg) const;

private:
    // Disallow copying/moving
    ThreadTeam(ThreadTeam& other)                = delete;
    ThreadTeam(const ThreadTeam& other)          = delete;
    ThreadTeam(ThreadTeam&& other)               = delete;
    ThreadTeam& operator=(ThreadTeam& rhs)       = delete;
    ThreadTeam& operator=(const ThreadTeam& rhs) = delete;
    ThreadTeam& operator=(ThreadTeam&& rhs)      = delete;

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
    unsigned int   N_wait_;      /*!< No data items at last transition.
                                  *   Waiting for transitionThread_. */
    unsigned int   N_comp_;      /*!< Found data item at last transition and
                                  *   currently applying action to item.
                                  *   It will transition
                                  *   (computingFinished event) when it
                                  *   finishes that work. */
    unsigned int   N_terminate_; /*!< Number of threads that are terminating
                                  *   or that have terminated. */

    std::queue<std::shared_ptr<DataItem>>   queue_;  /*!< Internal data item queue to which the
                                                      * team's actions need to be applied. */

    //***** Data members not directly related to state
    // Calling code could set the number of Idle threads by calling startCycle()
    // and increaseThreadCount().  However, there is a non-zero delay for the
    // threads to transition to active.
    //
    // If calling code calls increaseThreadCount(), we need to throw an error if
    // the increased count would exceed the number of Idle threads in the team.
    // Therefore, we cannot rely on the number of actual Idle threads to do this
    // error checking.  This tracks the number of Idle threads that will be
    // activated but that have not yet received the signal to activate.
    unsigned int      N_to_activate_;

    unsigned int      nMaxThreads_;        //!< Number of threads in team won't exceed this
    unsigned int      id_;                 //!< (Hopefully) unique ID for team
    std::string       hdr_;                //!< Short name of team for logging
    std::string       actionName_;         //!< Short name of action for logging

    pthread_attr_t    attr_;               //!< All threads setup with this attribute
    pthread_mutex_t   teamMutex_;          //!< Use to access members
    pthread_cond_t    allActivated_;       /*!< Emitted when a thread activates
                                            *   and determines that there are no more
                                            *   pending activations */
    pthread_cond_t    threadStarted_;      //!< Each thread emits this signal upon starting
    pthread_cond_t    activateThread_;     //!< Ask idling threads to transition state
    pthread_cond_t    transitionThread_;   //!< Ask waiting threads to transition state
    pthread_cond_t    threadTerminated_;   //!< Each thread emits this signal upon termination
    pthread_cond_t    unblockWaitThread_;  //!< Wake single thread blocked by calling wait()

    ACTION_ROUTINE    actionRoutine_;      /*!< Computational routine to be applied to
                                            *   all data items enqueued with team*/

    // Keep track of when wait() is blocking and when it is released
    bool              isWaitBlocking_;     //!< Only a single thread can be blocked 
};

}

#endif

