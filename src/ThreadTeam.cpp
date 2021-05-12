/*
 * \file    ThreadTeam.cpp
 */

#include "ThreadTeam.h"

#include <sys/time.h>
#include <iostream>
#include <stdexcept>

#include "OrchestrationLogger.h"
#include "ThreadTeamState.h"
#include "ThreadTeamIdle.h"
#include "ThreadTeamTerminating.h"
#include "ThreadTeamRunningOpen.h"
#include "ThreadTeamRunningClosed.h"
#include "ThreadTeamRunningNoMoreWork.h"

namespace orchestration {

// TODO:  The pthread implementation needs to be cleaned up seriously.  See
// TODOs below.
// TODO:  Come up with a good error/exception handling scheme.  The exceptions
// being thrown by the worker threads cannot be caught by the lead thread, which
// will lead to hard faults that cannot be traced easily.

/**
 * Instantiate a thread team that, at any point in time, can have no more than
 * nMaxThreads threads in existence.
 *
 * This routine initializes the state of the team in IDLE with
 *  - no threads waiting, computing, or terminating,
 *  - all nMaxThreads threads Idling, and
 *  - no data items in the queue.
 *
 * \param  nMaxThreads The maximum permissible number of threads in the team.
 *                     Zero or one thread is considered to be a logical error.
 * \param  id          A unique thread team ID for debug use.
 */
ThreadTeam::ThreadTeam(const unsigned int nMaxThreads,
                       const unsigned int id)
    : state_(nullptr),
      stateIdle_(nullptr),
      stateTerminating_(nullptr),
      stateRunOpen_(nullptr),
      stateRunClosed_(nullptr),
      stateRunNoMoreWork_(nullptr),
      N_idle_(0),
      N_wait_(0),
      N_comp_(0),
      N_terminate_(0),
      queue_(),
      N_to_activate_(0),
      nMaxThreads_(nMaxThreads),
      id_(id),
      hdr_("No Header Yet"),
      actionName_("No Action Yet"),
      actionRoutine_(nullptr),
      isWaitBlocking_(false)
{
    hdr_ = "Thread Team " + std::to_string(id_);
    
    if (nMaxThreads_ <= 1) {
        std::string   msg("[ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tTeams must have at least two threads";
        throw std::logic_error(msg);
    }

    // Initialize mutex before creating other states in case they need it 
    pthread_mutex_init(&teamMutex_, NULL);

    //***** INSTANTIATE EXTENDED FINITE STATE MACHINE STATE OBJECTS
    stateIdle_ = new ThreadTeamIdle(this); 
    if (!stateIdle_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(ThreadTeamMode::IDLE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateTerminating_ = new ThreadTeamTerminating(this); 
    if (!stateTerminating_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(ThreadTeamMode::TERMINATING);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunOpen_ = new ThreadTeamRunningOpen(this); 
    if (!stateRunOpen_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(ThreadTeamMode::RUNNING_OPEN_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunClosed_ = new ThreadTeamRunningClosed(this); 
    if (!stateRunClosed_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(ThreadTeamMode::RUNNING_CLOSED_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunNoMoreWork_ = new ThreadTeamRunningNoMoreWork(this); 
    if (!stateRunNoMoreWork_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(ThreadTeamMode::RUNNING_NO_MORE_WORK);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    pthread_mutex_lock(&teamMutex_);

    // TODO: Do we need to set more attributes?
    // TODO: Are the detached threads being handled appropriately so that
    //       we don't have any resource loss?
    pthread_attr_init(&attr_);
    pthread_attr_setdetachstate(&attr_, PTHREAD_CREATE_DETACHED);

    pthread_cond_init(&allActivated_, NULL);
    pthread_cond_init(&threadStarted_, NULL);
    pthread_cond_init(&activateThread_, NULL);
    pthread_cond_init(&transitionThread_, NULL);
    pthread_cond_init(&threadTerminated_, NULL);
    pthread_cond_init(&unblockWaitThread_, NULL);

    //***** SETUP EXTENDED FINITE STATE MACHINE IN INITIAL STATE
    // Setup before creating threads, which need to know the state
    // - IDLE with all threads in Idle and no pending dataItems
    N_idle_        = 0;
    N_wait_        = 0;
    N_comp_        = 0;
    N_terminate_   = 0;
    N_to_activate_ = 0;
    if (!queue_.empty()) {
        std::string  msg = printState_NotThreadsafe("ThreadTeam", 0,
                           "Data item queue is not empty");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(msg);
    }
    // Setup manually since we aren't transitioning yet
    state_ = stateIdle_;

    int rc = 0;
    pthread_t   threads[nMaxThreads_];
    ThreadData  threadData[nMaxThreads_];
    for (unsigned int i=0; i<nMaxThreads_; ++i) {
        threadData[i].tId  = i;
        threadData[i].team = this;
        rc = pthread_create(&threads[i], &attr_, *threadRoutine,
                            reinterpret_cast<void*>(&(threadData[i])));
        if (rc != 0) {
            std::string  msg = printState_NotThreadsafe("ThreadTeam", i,
                               "Unable to create thread");
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(msg);
        }
    }

    // Wait until all threads have started running their routine and are Idle
    struct timeval    now;
    struct timespec   waitAbsTime;
    while (N_idle_ < nMaxThreads_) {
        gettimeofday(&now, NULL);
        waitAbsTime.tv_sec  = now.tv_sec;
        waitAbsTime.tv_nsec = now.tv_usec * 1000;
        waitAbsTime.tv_sec += THREAD_START_STOP_TIMEOUT_SEC;

        rc = pthread_cond_timedwait(&threadStarted_, &teamMutex_, &waitAbsTime);
        if (rc == ETIMEDOUT) {
            std::string  msg = printState_NotThreadsafe("ThreadTeam", 0,
                               "Timeout on threads starting");
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(msg);
        }
    }
    N_to_activate_ = 0;

    unsigned int N_total = N_idle_ + N_wait_ + N_comp_;
    if (N_terminate_ != 0) {
        std::string  errMsg = printState_NotThreadsafe(
            "ThreadTeam", 0, "N_terminate_ not zero");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (N_total != nMaxThreads_) {
        std::string  errMsg = printState_NotThreadsafe(
            "ThreadTeam", 0, "Inconsistent thread counts");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = printState_NotThreadsafe("ThreadTeam", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    msg =   "[" + hdr_ + "] Team initialized in state " 
          + getModeName(state_->mode()) + " with "
          + std::to_string(N_idle_) + " threads idling";
    Logger::instance().log(msg);

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Destroy the thread team.  This routine will request that all threads
 * terminate first.  It cannot, however, request this of Computing threads, but
 * will wait for them to finish their dataItem.
 */
ThreadTeam::~ThreadTeam(void) {
    pthread_mutex_lock(&teamMutex_);

    // TODO: dequeue all items explicitly if queue not empty?  Definitely
    // necessary if the items in the queue are pointers to dynamically-allocated
    // memory.
    // TODO: Print warning messages if termination is not happening under normal
    // conditions so that clients can detect logical errors on their part.
    try {
        std::string  msg = setMode_NotThreadsafe(ThreadTeamMode::TERMINATING);
        if (msg != "") {
            std::string  errMsg = printState_NotThreadsafe("~ThreadTeam", 0, msg);
            std::cerr << errMsg << std::endl;
        } else {
            // We cannot assume that the team is in the nice Idle state.
            // Rather, it could be destroyed due to a runtime error
            //
            // Tell all waiting and idling threads to terminate.
            // Computing threads should figure this out once they have finished applying
            // their action to their current data item.
            N_terminate_ = 0;
            N_to_activate_ = N_idle_;
            pthread_cond_broadcast(&activateThread_);
            pthread_cond_broadcast(&transitionThread_);

            int               rc = 0;
            struct timeval    now;
            struct timespec   waitAbsTime;
            while (N_terminate_ < nMaxThreads_) {
                gettimeofday(&now, NULL);
                waitAbsTime.tv_sec  = now.tv_sec;
                waitAbsTime.tv_nsec = now.tv_usec * 1000;
                waitAbsTime.tv_sec += THREAD_START_STOP_TIMEOUT_SEC;

                rc = pthread_cond_timedwait(&threadTerminated_, &teamMutex_, &waitAbsTime);
                if (rc == ETIMEDOUT) {
                    std::string  errMsg = printState_NotThreadsafe("~ThreadTeam", 0,
                                       "Timeout on threads terminating");
                    std::cerr << errMsg << std::endl;
                    break;
                }
            }
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    std::string msg =   "[" + hdr_ + "] " 
                      + std::to_string(nMaxThreads_) + " Threads terminated";
    Logger::instance().log(msg);

    pthread_cond_destroy(&unblockWaitThread_);
    pthread_cond_destroy(&threadTerminated_);
    pthread_cond_destroy(&transitionThread_);
    pthread_cond_destroy(&activateThread_);
    pthread_cond_destroy(&threadStarted_);
    pthread_cond_destroy(&allActivated_);

    pthread_attr_destroy(&attr_);

    pthread_mutex_unlock(&teamMutex_);

    pthread_mutex_destroy(&teamMutex_);

    state_ = nullptr;
    if (stateIdle_) {
        delete stateIdle_;
        stateIdle_ = nullptr;
    }
    if (stateTerminating_) {
        delete stateTerminating_;
        stateTerminating_ = nullptr;
    }
    if (stateRunOpen_) {
        delete stateRunOpen_;
        stateRunOpen_ = nullptr;
    }
    if (stateRunClosed_) {
        delete stateRunClosed_;
        stateRunClosed_ = nullptr;
    }
    if (stateRunNoMoreWork_) {
        delete stateRunNoMoreWork_;
        stateRunNoMoreWork_ = nullptr;
    }

    msg = "[" + hdr_ + "] Team destroyed";
    Logger::instance().log(msg);
}

/**
 * Obtain the name of the given mode as a string.  This method does not access
 * resources and therefore can be called without acquiring the team's mutex.
 *
 * \param   mode - The enum value of the mode.
 * \return  The name
 */
std::string ThreadTeam::getModeName(const ThreadTeamMode mode) const {
    std::string   modeName("");

    switch(mode) {
    case ThreadTeamMode::IDLE:
        modeName = "Idle";
        break;
    case ThreadTeamMode::TERMINATING:
        modeName = "Terminating";
        break;
    case ThreadTeamMode::RUNNING_OPEN_QUEUE:
        modeName = "Running & Queue Open";
        break;
    case ThreadTeamMode::RUNNING_CLOSED_QUEUE:
        modeName = "Running & Queue Closed";
        break;
    case ThreadTeamMode::RUNNING_NO_MORE_WORK:
        modeName = "Running & No More Work";
        break;
    default:
        std::string msg("ThreadTeam::getModeName] ");
        msg += hdr_;
        msg += "\n\tInvalid state mode ";
        msg += std::to_string(static_cast<int>(mode));
        throw std::logic_error(msg);
    }

    return modeName;
}

/**
 * Set the mode of the EFSM to the given mode.  This internal method is usually
 * called as part of a state transition.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \param    nextMode - The next mode as an enum value
 * \return   An empty string if the mode was set successfully, an error
 *           statement otherwise.
 */
std::string ThreadTeam::setMode_NotThreadsafe(const ThreadTeamMode nextMode) {
    std::string    errMsg("");

#ifdef DEBUG_RUNTIME
    ThreadTeamMode  currentMode = state_->mode();
#endif

    switch(nextMode) {
    case ThreadTeamMode::IDLE:
        state_ = stateIdle_;
        break;
    case ThreadTeamMode::TERMINATING:
        state_ = stateTerminating_;
        break;
    case ThreadTeamMode::RUNNING_OPEN_QUEUE:
        state_ = stateRunOpen_;
        break;
    case ThreadTeamMode::RUNNING_CLOSED_QUEUE:
        state_ = stateRunClosed_;
        break;
    case ThreadTeamMode::RUNNING_NO_MORE_WORK:
        state_ = stateRunNoMoreWork_;
        break;
    default:
        return ("Unknown ThreadTeam mode " + getModeName(nextMode));
    }

    if (!state_) {
        return printState_NotThreadsafe("setMode_NotThreadsafe", 0,
               getModeName(nextMode) + " instance is NULL");
    } 
    std::string   msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        return msg;
    } else if (nextMode != state_->mode()) {
        msg  = "EFSM in mode " + getModeName(state_->mode());
        msg += " instead of intended mode " + getModeName(nextMode);
        return printState_NotThreadsafe("setMode_NotThreadsafe", 0, msg);
    }

#ifdef DEBUG_RUNTIME
    msg = "[" + hdr_ + "] Transitioned from "
          + getModeName(currentMode) + " to " + getModeName(state_->mode());
    Logger::instance().log(msg);
#endif

    return errMsg;
}

/**
 * Obtain the total number of threads that the thread team may contain at any
 * point in time.
 *
 * \return The number of threads.
 */
unsigned int ThreadTeam::nMaximumThreads(void) const {
    // This variable is set at instantiation and should not change after that.
    // Therefore this routine does not need to acquire the mutex to access it.
    return nMaxThreads_;
}

/**
 * Obtain the current mode of the team.
 *
 * \return The mode as an enum
 */
ThreadTeamMode ThreadTeam::mode(void) {
    pthread_mutex_lock(&teamMutex_);

    if (!state_) {
        std::string  errMsg("ThreadTeam::mode] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    ThreadTeamMode theMode = state_->mode();

    pthread_mutex_unlock(&teamMutex_);

    return theMode;
}

/**
 * Obtain the current mode of the team.
 *
 * \return The mode as an enum
 */
void ThreadTeam::stateCounts(unsigned int* N_idle,
                                 unsigned int* N_wait,
                                 unsigned int* N_comp,
                                 unsigned int* N_dataItem) {
    pthread_mutex_lock(&teamMutex_);

    *N_idle = N_idle_;
    *N_wait = N_wait_;
    *N_comp = N_comp_;
    *N_dataItem = queue_.size();

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Indicate to the thread team that it may activate the given number of Idle
 * threads so that they may help execute the current execution cycle's action.
 *
 * \param nThreads - The number of Idle threads to activate.
 */
void ThreadTeam::increaseThreadCount(const unsigned int nThreads) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("increaseThreadCount", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("increaseThreadCount", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (nThreads == 0) {
        errMsg = printState_NotThreadsafe("increaseThreadCount", 0,
                 "No sense in increasing by zero threads");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (nThreads > (N_idle_ - N_to_activate_)) {
        msg  = "nThreads (";
        msg += std::to_string(nThreads);
        msg += ") exceeds the number of threads available for activation";
        errMsg = printState_NotThreadsafe("increaseThreadCount", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = state_->increaseThreadCount_NotThreadsafe(nThreads);
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Start an execution cycle that will apply the given action to all data items
 * subsequently given to the team using the enqueue() method.
 * 
 * An execution cycle can begin with zero threads with the understanding that
 * the team will either never receive data items or that it will have threads
 * activated at a later time by calls to increaseThreadCount or by a thread
 * publisher.
 *
 * \param    action - an object that encapsulates the action to be applied by
 *                    activated threads to all data items enqueued with the team.
 * \param    teamName - a name to assign to the team that will be used for
 *                      logging the team during this execution cycle.
 * \param    waitForThreads - a testing/timing flag.  If true, then this member
 *                            function will block until the correct number of
 *                            threads have been successfully activated.
 */
void ThreadTeam::startCycle(const RuntimeAction& action,
                            const std::string& teamName,
                            const bool waitForThreads) {
    std::string   logMsg;
    std::string   dataType;
    std::string   nItems;
    switch (action.teamType) {
    case ThreadTeamDataType::TILE:
        dataType = "Tile";
        nItems = "n/a";
        break;
    case ThreadTeamDataType::BLOCK:
        dataType = "Block";
        nItems = "n/a";
        break;
    case ThreadTeamDataType::SET_OF_TILES:
        dataType = "Packet of Tiles";
        nItems = std::to_string(action.nTilesPerPacket);
        break;
    case ThreadTeamDataType::SET_OF_BLOCKS:
        dataType = "Packet of Blocks";
        nItems = std::to_string(action.nTilesPerPacket);
        break;
    case ThreadTeamDataType::OTHER:
        dataType = "Unknown";
        nItems = std::to_string(action.nTilesPerPacket);
        break;
    };
//    logMsg =   "[" + hdr_ + "] Assigned action " + action.name + "\n";
//    logMsg += "\t\t\t\tData type        " + dataType + "\n";
//    logMsg += "\t\t\t\tN Items/Packet   " + nItems + "\n";
//    logMsg += "\t\t\t\tN Threads        " + std::to_string(action.nInitialThreads);
//    Logger::instance().log(logMsg);

    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("startCycle", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("startCycle", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (action.nInitialThreads > N_idle_) {
        // Derived classes that implement startCycle should account for nonzero
        // N_to_activate_
        std::string  msg  = "nInitialThreads (";
        msg += std::to_string(action.nInitialThreads);
        msg += ") exceeds the number of threads available for activation";
        errMsg = printState_NotThreadsafe("startCycle", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (!action.routine) {
        errMsg = printState_NotThreadsafe("startCycle", 0,
                 "null action routine funtion pointer given");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = state_->startCycle_NotThreadsafe(action, teamName);
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    // It is intended that this only be used during timing tests when trying to
    // estimate the overhead of winding up and down a thread team with a given
    // number of activated threads
    if (waitForThreads) {
        while (N_to_activate_ > 0) {
            pthread_cond_wait(&(allActivated_), &(teamMutex_));
        }
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Indicate to the thread team that no more data items will be enqueued with the 
 * team during the present execution cycle, which will end once the team's
 * action has been applied to all data items presently in the queue (if any).
 *
 * \param publisher - a pointer to the data publisher that is calling this
 *                    function.  Passing a null pointer is only valid if the
 *                    calling code is an action parallel distributor.  If this
 *                    is the case, it is a logical error for the object to have
 *                    another data publisher.
 */
void ThreadTeam::closeQueue(const RuntimeElement* publisher) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("closeQueue", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("closeQueue", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    bool readyToClose = true;
    if (!publisher) {
        if (calledCloseQueue_.size() != 0) {
            msg = "If publisher is distributor, no other publisher allowed";
            errMsg = printState_NotThreadsafe("closeQueue", 0, msg);
            pthread_mutex_unlock(&teamMutex_);
            throw std::logic_error(errMsg);
        }

        // can proceed with closeQueue of subscriber
    } else {
        calledCloseQueue_.at(publisher) = true;
        for (auto& kv : calledCloseQueue_) {
            readyToClose = readyToClose && kv.second;
        }
    }

    if (readyToClose) {
        errMsg = state_->closeQueue_NotThreadsafe();
        if (errMsg != "") {
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(errMsg);
        }
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Give the thread team a data item on which it should apply the current
 * execution cycle's action.
 *
 * \param   dataItem - the data item
 */
void ThreadTeam::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string   errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("enqueue", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("enqueue", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    errMsg = state_->enqueue_NotThreadsafe(std::move(dataItem));
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Block a single calling thread until the team finishes executing its current
 * execution cycle.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable isWaitBlocking_ was
 * not included in the definition of the EFSM.
 */
void ThreadTeam::wait(void) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string   errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("wait", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("wait", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (isWaitBlocking_) {
        errMsg = printState_NotThreadsafe("wait", 0,
                 "A thread has already called wait");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    ThreadTeamMode  mode = state_->mode();
    if (mode == ThreadTeamMode::TERMINATING) {
        std::string  errMsg = printState_NotThreadsafe("wait", 0,
                              "Cannot call wait when terminating");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (mode == ThreadTeamMode::IDLE) {
        // Calling wait on a ThreadTeam that is Idle seems like a logic error.
        // However, it could be that a team finishes its job and transition to
        // Idle before a calling thread got a chance to call wait().  Therefore,
        // this method is a no-op so that it won't block.
#ifdef DEBUG_RUNTIME
        std::string msg = "[Client Thread] Called no-op wait on " 
                          + hdr_ + " team (Idle)";
        Logger::instance().log(msg);
#endif
    } else {
        isWaitBlocking_ = true;

#ifdef DEBUG_RUNTIME
        msg = "[Client Thread] Waiting on " + hdr_ + " team - " + getModeName(mode);
        Logger::instance().log(msg);
#endif

        pthread_cond_wait(&unblockWaitThread_, &teamMutex_);
        if (state_->mode() != ThreadTeamMode::IDLE) {
            std::string  msg = "Client thread unblocked with team in mode ";
            msg += getModeName(state_->mode());
            std::string  errMsg = printState_NotThreadsafe("wait", 0, msg);
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(errMsg);
        }

#ifdef DEBUG_RUNTIME
        msg = "[Client Thread] Received unblockWaitSignal for " + hdr_ + " team";
        Logger::instance().log(msg);
#endif

        isWaitBlocking_ = false;
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Attach given thread team as a thread subscriber.  Therefore, this converts
 * the calling object into a thread publisher.  A thread team shall be able
 * to attach to any other thread team as a subscriber regardless of the data
 * types of both teams. 
 *
 * It is a logic error
 *   - to attach a team to itself
 *   - to attach when a team is already attached.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 *
 * \param  receiver - the team to which thread transitions to Idle shall by
 *                    published.
 */
std::string ThreadTeam::attachThreadReceiver(RuntimeElement* receiver) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != ThreadTeamMode::IDLE) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "A team can only be attached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = RuntimeElement::attachThreadReceiver(receiver);
    if (errMsg != "") {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0, errMsg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);

    return "";
}

/**
 * Detach the thread subscriber so that the calling object is no longer a thread
 * publisher.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 */
std::string ThreadTeam::detachThreadReceiver(void) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != ThreadTeamMode::IDLE) {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0,
                 "A team can only be detached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = RuntimeElement::detachThreadReceiver();
    if (errMsg != "") {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0, errMsg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);

    return "";
}

/**
 * Register given thread team as a data subscriber.  Therefore, this converts
 * the calling object into a data publisher.  A data publisher and data
 * subscriber must have the same same data type.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 *
 * \param  receiver - the team to which data items shall be published.
 */
std::string ThreadTeam::attachDataReceiver(RuntimeElement* receiver) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("attachDataReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("attachDataReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != ThreadTeamMode::IDLE) {
        errMsg = printState_NotThreadsafe("attachDataReceiver", 0,
                 "A team can only be attached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = RuntimeElement::attachDataReceiver(receiver);
    if (errMsg != "") {
        errMsg = printState_NotThreadsafe("attachDataReceiver", 0, errMsg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);

    return "";
}

/**
 * Detach the data subscriber so that the calling object is no longer a data
 * publisher.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 */
std::string ThreadTeam::detachDataReceiver(void) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("detachDataReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("detachDataReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != ThreadTeamMode::IDLE) {
        errMsg = printState_NotThreadsafe("detachDataReceiver", 0,
                 "A team can only be detached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = RuntimeElement::detachDataReceiver();
    if (errMsg != "") {
        errMsg = printState_NotThreadsafe("detachDataReceiver", 0, errMsg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);

    return "";
}

/**
 * Obtain a state snapshot of the team for logging/debugging.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \param method - the name of the method that is requesting the state info
 * \param tId    - the unique thread ID of the calling thread
 * \param msg    - a error message to insert in the snapshot
 * \return The snapshot as a string.
 */
std::string  ThreadTeam::printState_NotThreadsafe(const std::string& method, 
                                                  const unsigned int tId,
                                                  const std::string& msg) const {
    // TODO: Print thread subscriber and data subscriber IDs
    //       Need to expand error handling to level of whole runtime so
    //       so that the snapshot is the team connectivity
    //       as well as the state of each team.
    std::string    state("");
    state += "[ThreadTeam::" + method + "] ";
    state += hdr_ + "/Thread " + std::to_string(tId);
    state += "\n\tContext - " + msg;
    state += "\n\tThread team state snapshot";
    state += "\n\t------------------------------------------------------------";
    state += "\n\tMode\t\t\t\t\t\t\t"             + getModeName(state_->mode());
    state += "\n\tN threads in team\t\t\t\t"      + std::to_string(nMaxThreads_);
    state += "\n\tN threads actually Idle\t\t\t"  + std::to_string(N_idle_);
    state += "\n\tN threads Waiting\t\t\t\t"      + std::to_string(N_wait_);
    state += "\n\tN threads Computing\t\t\t\t"    + std::to_string(N_comp_);
    state += "\n\tN threads Terminating\t\t\t"    + std::to_string(N_terminate_);
    state += "\n\tN data items in queue\t\t\t\t"  + std::to_string(queue_.size());
    state += "\n\tN threads pending activation\t" + std::to_string(N_to_activate_);

    return state;
}

/**
 * This is the routine that each thread in the team executes as it moves
 * through the thread states Idle, Waiting, Computing, and Terminating in accord
 * with the team's mode, the queue contents and the state of the other threads
 * in the team.
 * 
 * \todo Check if some errors could be thrown during destruction.  If so, do we
 *       accept such ugly failure handling or device a better logging/error
 *       handling scheme.
 *
 * \param  varg - a void pointer to the thread's ThreadData initialization data
 * \return nullptr
 */
void* ThreadTeam::threadRoutine(void* varg) {
    pthread_detach(pthread_self());

    ThreadData* data = reinterpret_cast<ThreadData*>(varg);
    if (!data) {
        pthread_exit(NULL);
        throw std::runtime_error("[ThreadTeam::threadRoutine] "
                                 "Null thread data pointer");
    }

    unsigned int tId = data->tId;
    ThreadTeam* team = data->team;
    if (!team) {
        pthread_exit(NULL);
        throw std::runtime_error("[ThreadTeam::threadRoutine] "
                                 "Null thread team pointer");
    }

    // The loop structure is as follows:
    // - A thread that is executing, is inside the loop, and has the mutex
    //   enters into a branch that corresponds to the current state.  While the
    //   thread is searching for its branch, the thread is understood to be
    //   transitioning and therefore not in a final state.
    // - the thread increments the count for the state into which it is
    //   transitioning
    // - the thread confirms that N_i + N_w + N_c + N_t = N_max is satisfied
    // - the thread performs any output that is associated with the final mode
    //   of the transition and independent of the starting mode of the
    //   transition.
    // - the thread waits on a signal and therefore relinquishes the mutex.
    //   The transition has finished.
    //   Note that for a computing thread "wait" means block while it applies
    //   the action to its data item.  The termination of the work is the
    //   "signal" that the thread sends to itself to transition.
    // - upon receiving a signal to transition and therefore recovering the
    //   mutex, the thread executes output that is associated with the current
    //   mode and independent of the next mode
    // - the thread decrements the counter for the current mode
    // - the thread then loops around to repeat the process.
    // TODO: Does this EFSM have outputs that depend on both the current and
    //       next mode?
    //
    // Put mutex outside of the loop so that transitions happen with one single
    // acquisition of the mutex.  
    //
    // Example: A thread receives the activateThread_ signal and therefore 
    //          acquires the mutex.  Within its current branch, it decrements
    //          N_idle_, loops around, and enters the next branch.  If it
    //          determines that it should wait, it then increments N_w_.  If the
    //          mutex were released at the end of the loop body, then there
    //          would be some time during which N_i + N_w + N_c + N_t != N_max.
    pthread_mutex_lock(&(team->teamMutex_));

    std::string      errMsg("");
    ThreadTeamMode   mode             = ThreadTeamMode::IDLE;
    bool             isThreadStarting = true;
    unsigned int     N_Q              = 0;
    unsigned int     N_total          = 0;
    std::shared_ptr<DataItem>   dataItem{};
    while (true) {
        if ((dataItem.get() != nullptr) || (dataItem.use_count() != 0)) {
            std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "dataItem should be NULL at top of loop body");
            std::cerr << msg << std::endl;
            pthread_mutex_unlock(&(team->teamMutex_));
            throw std::logic_error(msg);
        }

        mode = team->state_->mode();
        // TODO: The amount of time that a thread locks the mutex will grow as
        // the number of data items in the Q grows due to call like this.  A
        // possible, but ugly, optimization would be to manage N_Q without
        // getting it from the queue_.size().
        // NOTE: It could be that the stdlib is already doing something like
        // this, but this would likely be implementation-dependent.
        N_Q = team->queue_.size();

        // Finish transition, wait for event, do output, and loop back
        if (mode == ThreadTeamMode::TERMINATING) {
            // 
            //<----------- TRANSITION TO TERMINATING & STOP THREADS ----------->
            //

            // Overflow would be caught below by the check on N_total.
            team->N_terminate_ += 1;

            N_total =   team->N_idle_ + team->N_wait_
                      + team->N_comp_ + team->N_terminate_;
            if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            }

#ifdef DEBUG_RUNTIME
            std::string msg =   "[" + team->hdr_ + " / Thread "
                              + std::to_string(tId) + "] Terminated - " 
                              + std::to_string(team->N_terminate_)
                              + " terminated out of "
                              + std::to_string(team->nMaxThreads_);
            Logger::instance().log(msg);
#endif

            // Inform team that thread has terminated & terminate
            pthread_cond_signal(&(team->threadTerminated_));
            pthread_mutex_unlock(&(team->teamMutex_));
            pthread_exit(NULL);
            return nullptr;
        } else if (    (mode == ThreadTeamMode::IDLE)
                   ||  (mode == ThreadTeamMode::RUNNING_NO_MORE_WORK)
                   || (   (mode == ThreadTeamMode::RUNNING_CLOSED_QUEUE) 
                       && (N_Q == 0)) ) {
            // 
            //<--------------------- TRANSITION TO IDLE --------------------->
            //
            // Finish a thread transition by setting thread to Idle

            // Overflow would be caught below by the check on N_total.
            team->N_idle_ += 1;

            if (isThreadStarting) {
                // If starting thread team, we cannot yet check that
                // N_total law is satisfied - the constructor confirms this
                isThreadStarting = false;
                pthread_cond_signal(&(team->threadStarted_));
            } else {
                N_total =   team->N_idle_
                          + team->N_wait_
                          + team->N_comp_;
                if (team->N_terminate_ != 0) {
                    std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, "N_terminate_ not zero");
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    // TODO: Which thread can catch this exception?
                    throw std::runtime_error(msg);
                } else if (N_total != team->nMaxThreads_) {
                    std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, "Inconsistent thread counts");
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }

                if (team->threadReceiver_) {
                    team->threadReceiver_->increaseThreadCount(1);
                }
            }

#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Transition to Idle"));
#endif

            // These two conditionals must appear in this order as the first
            // branch could make a change that gets caught by the second.
            // 
            // Since these are emitting events, only call these after updating
            // the state for this transition.
            if ((mode == ThreadTeamMode::RUNNING_CLOSED_QUEUE) && (N_Q == 0)) {
                mode = ThreadTeamMode::RUNNING_NO_MORE_WORK;
                errMsg = team->setMode_NotThreadsafe(mode);
                if (errMsg != "") {
                    std::string  msg = team->printState_NotThreadsafe(
                                             "threadRoutine", tId, errMsg);
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                pthread_cond_broadcast(&(team->transitionThread_));
            }
            // The job has been completely applied and the execution cycle
            // is over if this is the last thread to transition to Idle
            // Finalize for the end of the cycle
            if (   (mode == ThreadTeamMode::RUNNING_NO_MORE_WORK) 
                && (team->N_idle_ == team->nMaxThreads_)) {
                errMsg = team->setMode_NotThreadsafe(ThreadTeamMode::IDLE);
                if (errMsg != "") {
                    std::string  msg = team->printState_NotThreadsafe(
                                             "threadRoutine", tId, errMsg);
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                if (team->dataReceiver_) {
                    team->dataReceiver_->closeQueue(team);
                }

#ifdef DEBUG_RUNTIME
                Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                      "Sent unblockWaitThread signal"));
#endif

                // reset team name to generic name
                team->hdr_ = "Thread Team " + std::to_string(team->id_);
                team->actionRoutine_ = nullptr;

                // Make sure that this thread is ready to go Idle before
                // releasing the blocked thread
                pthread_cond_broadcast(&(team->unblockWaitThread_));
            }

            pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));
#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Activated"));
#endif

            if (team->N_to_activate_ <= 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, 
                    "Thread activated with N_to_activate_ zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            } else if (team->N_idle_ <= 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_idle unexpectedly zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }
            team->N_to_activate_ -= 1;
            team->N_idle_ -= 1;

            // Let startCycle/increaseThreadCount know that there are no more
            // pending activations
            if (team->N_to_activate_ == 0) {
                pthread_cond_signal(&(team->allActivated_));
            }
        } else if (N_Q == 0) {
            // 
            //<--------------------- TRANSITION TO WAIT --------------------->
            //
            // Should be in Running & Open

            // Overflow would be caught below by the check on N_total.
            team->N_wait_ += 1;

            N_total =   team->N_idle_
                      + team->N_wait_
                      + team->N_comp_;
            if (team->N_terminate_ != 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_terminate_ not zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            } else if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }

#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Transition to Waiting"));
#endif
            pthread_cond_wait(&(team->transitionThread_), &(team->teamMutex_));
#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Awakened"));
#endif

            if (team->N_wait_ <= 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_wait unexpectedly zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }
            team->N_wait_ -= 1;
        } else {
            // 
            //<--------------------- TRANSITION TO COMPUTING --------------------->
            //
            // Should be in Running & Open or Running & Closed with pending work

            // Overflow would be caught below by the check on N_total.
            team->N_comp_ += 1;

            N_total =   team->N_idle_
                      + team->N_wait_
                      + team->N_comp_;
            if (team->N_terminate_ != 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_terminate_ not zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            } else if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }

#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Transition to Computing"));
#endif

            // dataItem is assumed to be null at this point.
            // Move the data item from the queue to here
            // The item still in the queue will be nulled and will no longer
            // be sharing the data item.   This should, therefore, not change
            // the shared_ptr's counter.
            dataItem = std::move(team->queue_.front());
            // This pop should not decrease the internal counter of the shared_ptr
            team->queue_.pop();
            --N_Q;

#ifdef DEBUG_RUNTIME
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Dequeued dataItem"));
#endif

            // Since this emits events, only emit event after updating the mode
            // for this transition.
            //
            // No more pending tasks for this job (aside from the task
            // associated with the data item just popped).  Wake up all Waiting
            // threads so that they decide to go Idle
            // 
            // NOTE: This would be handled above, but check it here so that we
            // can release waiting threads as soon as possible and therefore
            // pass these resources to thread subscribers ASAP.
            if ((mode == ThreadTeamMode::RUNNING_CLOSED_QUEUE) && (N_Q == 0)) {
                errMsg = team->setMode_NotThreadsafe(
                               ThreadTeamMode::RUNNING_NO_MORE_WORK);
                if (errMsg != "") {
                    std::string  msg = team->printState_NotThreadsafe(
                                             "threadRoutine", tId, errMsg);
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                pthread_cond_broadcast(&(team->transitionThread_));
            }

            // Do work!
            //
            // The transition to computing is completed here as the computing
            // thread effectively "sleeps" in the sense that it is not playing
            // an active role in the ThreadTeam at this point. Rather it is
            // effectively waiting to receive the computationFinished event.
            pthread_mutex_unlock(&(team->teamMutex_));

            // The shared-ness of dataItem stays here, which means that the 
            // object pointed to cannot be destroyed until dataItem is reset.
            // Therefore, we can just pass the bare pointer and know
            // that actionRoutine_ can use it correctly.
            if ((dataItem.get() == nullptr) || (dataItem.use_count() <= 0)) {
                std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, "dataItem is unexpectedly NULLed");
                std::cerr << msg << std::endl;
                throw std::logic_error(msg);
            }
            team->actionRoutine_(tId, dataItem.get());

            // This is where computationFinished is "emitted" and the thread's
            // next transition begins.  A simple run-to-completion design would
            // acquire the mutex here before proceeding to transition or perform
            // outputs associated with the transition.  Indeed, this was how the
            // ThreadTeam was originally implemented.  See Technical
            // Specification 7.1.
            //
            // However, acquiring the mutex here can seriously impact
            // performance if this thread's team has a data subscriber.  In
            // particular, if the thread acquired the mutex here and enqueued
            // the data item with the subscriber, then it blocks until the
            // enqueue finishes all other threads in its team that are waiting
            // on the mutex.  Therefore, if the thread is blocked on the enqueue
            // call because the subscriber's mutex is not available or because
            // the enqueue takes a large amount of time, then the thread team
            // grinds to a halt.
            //
            // I have seen this happen when profiling the 3D Sedov/GPU problem
            // on Summit with NSight.  The issue was that the asynchronous
            // device-to-host and subsequent registering of a callback function
            // executed as part of the DataMover's enqueue method were slowed
            // down tremendously.  This occurred because the CUDA runtime
            // (10.1.243) appears to use pthreads and each CUDA runtime call
            // made by our runtime appears to acquire the same mutex before
            // carrying out the associated actions regardless of which CUDA
            // runtime function was called or with which stream the action
            // should be associated.  As a result, when these enqueue calls to
            // our runtime get stuck because the CUDA runtime mutex cannot be
            // accessed, the whole thread team can stall.
            //
            // The next block of code executes an output associated with the
            // thread's next transition.  Specifically, it
            // 1. enqueues the dataItem with the team's data subscriber if it
            // exists.
            // 2. Either passes the ownership of the data item to the data
            // subscriber or decrements the count of users of the shared_ptr.
            // Either way, the thread-private dataItem variable should be
            // nullified.
            //
            // CLAIM: Even though the computing thread's next transition begins
            // here and enqueuing the dataItem with the team's data subscriber
            // is an output associated with the transition, the code can do so
            // without acquiring the mutex and without violating Technical
            // Specification 7.1.  In other words, the eFSM will still function
            // correctly if this single aspect of the transition is run in
            // parallel with any number of possible transition of the eFSM.
            // While transitions are no longer atomic due to this change, only a
            // single thread is allowed to change the state at any time.
            //
            // Proof:
            // The pointer dataReceiver_ is a shared resource.  Note, however,
            // that Technical Specification 7.6 specifies that data
            // publisher/subscriber relationships can only change when the team
            // is in the Idle state.  However, the fact that this thread is
            // transitioning from computing implies that the team cannot be in
            // the Idle state.  Hence, no internal nor external code will change
            // dataReceiver_ and we are justified in using it without acquiring
            // the mutex.
            //
            // The data subscriber's enqueue method is thread-safe and its
            // successful execution does not depend on the state of this
            // thread's team.  Therefore, if two or more threads from this team
            // were to call enqueue at roughly the same time, the subscriber
            // will serialize access to the subscriber's resources.  In
            // addition, correct execution of computation by the calling thread
            // team will not be affected by the order in which enqueueing occurs
            // due to the fact that the task functions to be applied by the
            // team's pipeline will still be applied to each data item in the
            // correct order.  Therefore, protection of the shared resource is
            // managed by the resource itself and there is no possibility of a
            // race condition on calling enqueue.
            //
            // TODO: Do the technical specs or requirements require th addition
            // of a req/spec that states that the enqueue method of all runtime
            // elements must be thread safe and therefore manage access to
            // internal shared resources by external calls?
            //
            // dataItem is a thread-private variable.  This thread obtained the
            // shared_ptr stored in dataItem by removing it from the team's
            // shared queue.  In addition, N_Q was decremented and the mode
            // transitioned to RunningNoMoreWork if N_Q=0.  Therefore, this
            // thread is the only thread in the team that can interact with both
            // the shared_ptr and the resource that it points to (Spec 7.2) --- the
            // shared_ptr and its resource no longer exist at the level of the
            // team.  In other words, once the task function has been applied to
            // the data item by this thread, the thread can transfer the data
            // item to another runtime element and nullify its dataItem variable
            // with no possibility of a race condition on dataItem or the
            // resource that it points to.
            //
            // Note that the output performed here without acquiring the mutex
            // does not update the eFSM state.  In addition, correct execution
            // of the output does not depend on the team's state nor require
            // updating the state of the eFSM.  Therefore, any threads in
            // possession of the mutex at the time of executing this output are
            // still in control of the state.  Also, if one of these threads
            // transitions the state, correct execution of this output will not
            // be affected.  Hence all such threads and this computation thread
            // will execute correctly.

#ifdef DEBUG_RUNTIME
            // Since this is debug, allow for woefully poor performance.
            pthread_mutex_lock(&(team->teamMutex_));
            Logger::instance().log(team->printState_NotThreadsafe("threadRoutine", tId,
                                                                  "Finished computing"));
            pthread_mutex_unlock(&(team->teamMutex_));
#endif

            if (team->dataReceiver_) {
                // Move the data item along so that dataItem is null
                team->dataReceiver_->enqueue(std::move(dataItem));
            } else {
                // The data item is done.  Null dataItem so that the current
                // data item's resources can be released if this was the last
                // shared pointer to the data item.
                dataItem.reset();
            }
            if ((dataItem.get() != nullptr) || (dataItem.use_count() != 0)) {
                std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, "dataItem is not NULLed after computation");
                std::cerr << msg << std::endl;
                // NOTE: This must be commented out if the mutex acquisition
                // occurs directly after this code block.  Uncomment if the
                // mutex acquisition is moved above.
//                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::logic_error(msg);
            }

            pthread_mutex_lock(&(team->teamMutex_));

            if (team->N_comp_ <= 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_comp unexpectedly zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }
            team->N_comp_ -= 1;
        }
    }
    pthread_mutex_unlock(&(team->teamMutex_));

    std::string  msg = team->printState_NotThreadsafe("threadRoutine", tId,
                       "Thread unexpectedly reached end of routine");
    std::cerr << msg << std::endl;
    throw std::logic_error(msg);
}

}
