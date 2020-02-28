/*
 * \file    ThreadTeam.cpp
 */

#include "ThreadTeam.h"

#include <iostream>
#include <stdexcept>

#include "ThreadTeamIdle.h"
#include "ThreadTeamTerminating.h"
#include "ThreadTeamRunningOpen.h"
#include "ThreadTeamRunningClosed.h"
#include "ThreadTeamRunningNoMoreWork.h"

/**
 * Instantiate a thread team that, at any point in time, can have no more than
 * nMaxThreads threads in existence.
 *
 * This routine initializes the state of the team in MODE_IDLE with
 *  - no threads waiting, computing, or terminating,
 *  - all nMaxThreads threads Idling, and
 *  - no pending work.
 *
 * \param  nMaxThreads The maximum permissible number of threads in the team.
 *                     Zero or one thread is considered to be a logical error.
 * \param  id          A unique thread team ID for debug use.
 * \param  logFilename The file to which logging information is appended if the
 *                     code is built with VERBOSE.
 */
ThreadTeam::ThreadTeam(const unsigned int nMaxThreads,
                       const unsigned int id,
                       const std::string& logFilename)
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
      taskName_("No Task Yet"),
      taskFcn_(nullptr),
      threadReceiver_(nullptr),
      workReceiver_(nullptr),
      isWaitBlocking_(false),
      logFilename_(logFilename)
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
        msg += getModeName(MODE_IDLE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateTerminating_ = new ThreadTeamTerminating(this); 
    if (!stateTerminating_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(MODE_TERMINATING);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunOpen_ = new ThreadTeamRunningOpen(this); 
    if (!stateRunOpen_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(MODE_RUNNING_OPEN_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunClosed_ = new ThreadTeamRunningClosed(this); 
    if (!stateRunClosed_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(MODE_RUNNING_CLOSED_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunNoMoreWork_ = new ThreadTeamRunningNoMoreWork(this); 
    if (!stateRunNoMoreWork_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\n\tUnable to instantiate ";
        msg += getModeName(MODE_RUNNING_NO_MORE_WORK);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    pthread_mutex_lock(&teamMutex_);

    // TODO: Do we need to set more attributes?
    // TODO: Are the detached threads being handles appropriately so that
    //       we don't have any resource loss?
    pthread_attr_init(&attr_);
    pthread_attr_setdetachstate(&attr_, PTHREAD_CREATE_DETACHED);

    pthread_cond_init(&threadStarted_, NULL);
    pthread_cond_init(&activateThread_, NULL);
    pthread_cond_init(&transitionThread_, NULL);
    pthread_cond_init(&threadTerminated_, NULL);
    pthread_cond_init(&unblockWaitThread_, NULL);

    //***** SETUP EXTENDED FINITE STATE MACHINE IN INITIAL STATE
    // Setup before creating threads, which need to know the state
    // - MODE_IDLE with all threads in Idle and no pending work
    N_idle_        = 0;
    N_wait_        = 0;
    N_comp_        = 0;
    N_terminate_   = 0;
    N_to_activate_ = 0;
    if (!queue_.empty()) {
        std::string  msg = printState_NotThreadsafe("ThreadTeam", 0,
                           "Work queue is not empty");
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
    while (N_idle_ < nMaxThreads_) {
        // TODO: Timeout on this?
        pthread_cond_wait(&threadStarted_, &teamMutex_);
    }
    N_to_activate_ = 0;

    unsigned int N_total = N_idle_ + N_wait_ + N_comp_ + N_terminate_;
    if (N_total != nMaxThreads_) {
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

#ifdef VERBOSE
        logFile_.open(logFilename_, std::ios::out | std::ios::app);
        logFile_ << "[" << hdr_ << "] Team initialized in state " 
                 << getModeName(state_->mode())
                 << " with "
                 << N_idle_ << " threads idling\n";
        logFile_.close();
#endif

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Destroy the thread team.  This routine will request that all threads
 * terminate first.  It cannot, however, request this of Computing threads, but
 * will wait for them to finish their work.
 */
ThreadTeam::~ThreadTeam(void) {
    pthread_mutex_lock(&teamMutex_);

    // TODO: dequeue all items explicitly if queue not empty?  Definitely
    // necessary if the items in the queue are pointers to dynamically-allocated
    // memory.
    // TODO: Print warning messages if not in Idle?
    try {
        std::string  msg = setMode_NotThreadsafe(MODE_TERMINATING);
        if (msg != "") {
            std::string  errMsg = printState_NotThreadsafe("~ThreadTeam", 0, msg);
            std::cerr << errMsg << std::endl;
        } else {
            // We cannot assume that the team is in the nice Idle state.
            // Rather, it could be destroyed due to a runtime error
            //
            // Tell all waiting and idling threads to terminate.
            // Computing threads should figure this out once they have finished applying
            // their task to their current unit of work.
            N_terminate_ = 0;
            N_to_activate_ = N_idle_;
            pthread_cond_broadcast(&activateThread_);
            pthread_cond_broadcast(&transitionThread_);
            while (N_terminate_ < nMaxThreads_) {
                // TODO: Timeout on this?
                pthread_cond_wait(&threadTerminated_, &teamMutex_);
            }
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

#ifdef VERBOSE
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[" << hdr_ << "] " 
             << nMaxThreads_
             << " Threads terminated\n";
    logFile_.close();
#endif

    pthread_cond_destroy(&unblockWaitThread_);
    pthread_cond_destroy(&threadTerminated_);
    pthread_cond_destroy(&transitionThread_);
    pthread_cond_destroy(&activateThread_);
    pthread_cond_destroy(&threadStarted_);

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

#ifdef VERBOSE
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[" << hdr_ << "] Team destroyed\n";
    logFile_.close();
#endif
}

/**
 * Obtain the name of the given mode as a string.  This method does not access
 * resources and therefore can be called without acquiring the team's mutex.
 *
 * \param   mode - The enum value of the mode.
 * \return  The name
 */
std::string ThreadTeam::getModeName(const teamMode mode) const {
    std::string   modeName("");

    switch(mode) {
    case MODE_IDLE:
        modeName = "Idle";
        break;
    case MODE_TERMINATING:
        modeName = "Terminating";
        break;
    case MODE_RUNNING_OPEN_QUEUE:
        modeName = "Running & Queue Open";
        break;
    case MODE_RUNNING_CLOSED_QUEUE:
        modeName = "Running & Queue Closed";
        break;
    case MODE_RUNNING_NO_MORE_WORK:
        modeName = "Running & No More Work";
        break;
    default:
        std::string msg("ThreadTeam::getModeName] ");
        msg += hdr_;
        msg += "\n\tInvalid state mode ";
        msg += std::to_string(mode);
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
std::string ThreadTeam::setMode_NotThreadsafe(const teamMode nextMode) {
    std::string    errMsg("");

#ifdef VERBOSE
    teamMode  currentMode = state_->mode();
#endif

    switch(nextMode) {
    case MODE_IDLE:
        state_ = stateIdle_;
        break;
    case MODE_TERMINATING:
        state_ = stateTerminating_;
        break;
    case MODE_RUNNING_OPEN_QUEUE:
        state_ = stateRunOpen_;
        break;
    case MODE_RUNNING_CLOSED_QUEUE:
        state_ = stateRunClosed_;
        break;
    case MODE_RUNNING_NO_MORE_WORK:
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

#ifdef VERBOSE
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[" << hdr_ << "] Transitioned from "
             << getModeName(currentMode)
             << " to "
             << getModeName(state_->mode()) << std::endl;
    logFile_.close();
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
ThreadTeam::teamMode ThreadTeam::mode(void) {
    pthread_mutex_lock(&teamMutex_);

    if (!state_) {
        std::string  errMsg("ThreadTeam::mode] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    teamMode theMode = state_->mode();

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
                             unsigned int* N_work) {
    pthread_mutex_lock(&teamMutex_);

    *N_idle = N_idle_;
    *N_wait = N_wait_;
    *N_comp = N_comp_;
    *N_work = queue_.size();

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Indicate to the thread team that it may activate the given number of Idle
 * threads so that they may help execute the current execution cycle's task.
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
 * Start an execution cycle that will apply the given task to all tiles
 * subsequently given to the team using the enqueue() method.
 * 
 * An execution cycle can begin with zero threads with the understanding that
 * the team will either never receive work or that it will have threads
 * activated at a later time by calls to increaseThreadCount or by a thread
 * publisher.
 *
 * \param    fcn - a function pointer to the task to execute.
 * \param    nThreads - the number of Idle threads to immediately activate.
 * \param    teamName - a name to assign to the team that will be used for
 *                      logging the team during this execution cycle.
 * \param    taskName - a name to assign to the task that will be used for
 *                      logging the team during this execution cycle.
 */
void ThreadTeam::startTask(TASK_FCN* fcn,
                           const unsigned int nThreads,
                           const std::string& teamName, 
                           const std::string& taskName) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("startTask", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("startTask", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (nThreads > N_idle_) {
        // We don't need to consider N_to_activate_ here as those are 
        // considered in how many events to emit (See ThreadTeamRunningOpen))
        std::string  msg  = "nThreads (";
        msg += std::to_string(nThreads);
        msg += ") exceeds the number of threads available for activation";
        errMsg = printState_NotThreadsafe("startTask", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (!fcn) {
        errMsg = printState_NotThreadsafe("startTask", 0,
                 "null task funtion pointer given");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    errMsg = state_->startTask_NotThreadsafe(fcn, nThreads, teamName, taskName);
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Indicate to the thread team that no more units of work will be given to the
 * team during the present execution cycle.
 */
void ThreadTeam::closeTask(void) {
    pthread_mutex_lock(&teamMutex_);

    // Test conditions that should be checked regardless of team's current mode
    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("closeTask", 0, "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("closeTask", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    errMsg = state_->closeTask_NotThreadsafe();
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Give the thread team a unit of work on which it should apply the current
 * execution cycle's task.
 *
 * \todo    Need to figure out how to manage more than one unit of work.
 *          For a CPU-heavy task, it should receive tiles; a GPU-heavy
 *          task, a data packet of blocks.  Templates?
 *
 * \param   work - the unit of work.
 */
void ThreadTeam::enqueue(const int work) {
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

    errMsg = state_->enqueue_NotThreadsafe(work);
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

    teamMode  mode = state_->mode();
    if (mode == MODE_TERMINATING) {
        std::string  errMsg = printState_NotThreadsafe("wait", 0,
                              "Cannot call wait when terminating");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (mode == MODE_IDLE) {
        // Calling wait on a ThreadTeam that is Idle seems like a logic error.
        // However, it could be that a team finishes its task and transition to
        // Idle before a calling thread got a chance to call wait().  Therefore,
        // this method is a no-op so that it won't block.
#ifdef VERBOSE
        logFile_.open(logFilename_, std::ios::out | std::ios::app);
        logFile_ << "[Client Thread] Called no-op wait (Idle)\n";
        logFile_.close();
#endif
    } else {
        isWaitBlocking_ = true;

#ifdef VERBOSE
        logFile_.open(logFilename_, std::ios::out | std::ios::app);
        logFile_ << "[Client Thread] Waiting on team - "
                 << getModeName(mode) << std::endl;
        logFile_.close();
#endif

        pthread_cond_wait(&unblockWaitThread_, &teamMutex_);
        if (state_->mode() != MODE_IDLE) {
            std::string  msg = "Client thread unblocked with team in mode ";
            msg += getModeName(state_->mode());
            std::string  errMsg = printState_NotThreadsafe("wait", 0, msg);
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(errMsg);
        }

#ifdef VERBOSE
        logFile_.open(logFilename_, std::ios::out | std::ios::app);
        logFile_ << "[Client Thread] Received unblockWaitSignal\n";
        logFile_.close();
#endif

        isWaitBlocking_ = false;
    }

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Attach given thread team as a thread subscriber.  Therefore, this converts
 * the calling object into a thread publisher.
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
void ThreadTeam::attachThreadReceiver(ThreadTeam* receiver) {
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
    } else if (state_->mode() != MODE_IDLE) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "A team can only be attached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    if (!receiver) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "Null thread subscriber team given");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (receiver == this) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "Cannot attach team to itself");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (threadReceiver_) {
        errMsg = printState_NotThreadsafe("attachThreadReceiver", 0,
                 "A thread subscriber is already attached");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    threadReceiver_ = receiver;

    pthread_mutex_unlock(&teamMutex_);
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
void ThreadTeam::detachThreadReceiver(void) {
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
    } else if (state_->mode() != MODE_IDLE) {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0,
                 "A team can only be detached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    if (!threadReceiver_) {
        errMsg = printState_NotThreadsafe("detachThreadReceiver", 0,
                 "No thread subscriber attached");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    threadReceiver_ = nullptr;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Register given thread team as a work subscriber.  Therefore, this converts
 * the calling object into a work publisher.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 *
 * \param  receiver - the team to which units of work shall be published.
 */
void ThreadTeam::attachWorkReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != MODE_IDLE) {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0,
                 "A team can only be attached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    if (!receiver) {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0,
                 "Null work subscriber team given");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (receiver == this) {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0,
                 "Cannot attach team to itself");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    } else if (workReceiver_) {
        errMsg = printState_NotThreadsafe("attachWorkReceiver", 0,
                 "A work subscriber is already attached");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    workReceiver_ = receiver;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Detach the work subscriber so that the calling object is no longer a work
 * publisher.
 *
 * This routine is handled outside of the State design pattern as the mode
 * dependence is simple.  Also, the state variable publisher/not publisher was
 * not included in the definition of the EFSM.  Rather for simplicity, the
 * outputs use this information.
 */
void ThreadTeam::detachWorkReceiver(void) {
    pthread_mutex_lock(&teamMutex_);

    std::string    errMsg("");
    if (!state_) {
        errMsg = printState_NotThreadsafe("detachWorkReceiver", 0,
                 "state_ is NULL");
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        errMsg = printState_NotThreadsafe("detachWorkReceiver", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    } else if (state_->mode() != MODE_IDLE) {
        errMsg = printState_NotThreadsafe("detachWorkReceiver", 0,
                 "A team can only be detached in the Idle mode");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }
    
    if (!workReceiver_) {
        errMsg = printState_NotThreadsafe("detachWorkReceiver", 0,
                 "No work subscriber attached");
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error(errMsg);
    }

    workReceiver_ = nullptr;

    pthread_mutex_unlock(&teamMutex_);
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
    // TODO: Print thread subscriber and work subscriber IDs
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
    state += "\n\tN units of work in queue\t\t"   + std::to_string(queue_.size());
    state += "\n\tN threads pending activation\t" + std::to_string(N_to_activate_);
    state += "\n";

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
    //   the task to its unit of work.  The termination of the work is the
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

    std::string        errMsg("");
    teamMode           mode             = MODE_IDLE;
    bool               isThreadStarting = true;
    unsigned int       N_Q              = 0;
    unsigned int       work             = 0;
    unsigned int       N_total          = 0;
    while (true) {
        mode = team->state_->mode();
        N_Q = team->queue_.size();

        // Finish transition, wait for event, do output, and loop back
        if (mode == MODE_TERMINATING) {
            // 
            //<----------- TRANSITION TO TERMINATING & STOP THREADS ----------->
            //
            // TODO Check for overflow 
            team->N_terminate_ += 1;
            N_total =   team->N_idle_ + team->N_wait_
                      + team->N_comp_ + team->N_terminate_;
            if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            }

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << "[" << team->hdr_ << " / Thread " << tId << "] "
                           << "Terminated - " 
                           << team->N_terminate_
                           << " terminated out of "
                           << team->nMaxThreads_ << std::endl;
            team->logFile_.close();
#endif

            // Inform team that thread has terminated & terminate
            pthread_cond_signal(&(team->threadTerminated_));
            pthread_mutex_unlock(&(team->teamMutex_));
            pthread_exit(NULL);
            return nullptr;
        } else if (    (mode == MODE_IDLE)
                   ||  (mode == MODE_RUNNING_NO_MORE_WORK)
                   || ((mode == MODE_RUNNING_CLOSED_QUEUE) && (N_Q == 0)) ) {
            // 
            //<--------------------- TRANSITION TO IDLE --------------------->
            //

            // Finish a thread transition by setting thread to Idle
            // TODO Check for overflow 
            team->N_idle_ += 1;

            if (isThreadStarting) {
                // If starting thread team, we cannot yet check that
                // N_total law is satisfied - the constructor confirms this
                isThreadStarting = false;
                pthread_cond_signal(&(team->threadStarted_));
            } else {
                // Exclude N_terminate_ so that we also check that it is zero
                N_total =   team->N_idle_
                          + team->N_wait_
                          + team->N_comp_;
                if (N_total != team->nMaxThreads_) {
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

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Transition to Idle");
            team->logFile_.close();
#endif

            // These two conditionals must appear in this order as the first
            // branch could make a change that gets caught by the second.
            // 
            // Since these are emitting events, only call these after updating
            // the state for this transition.
            if ((mode == MODE_RUNNING_CLOSED_QUEUE) && (N_Q == 0)) {
                mode = MODE_RUNNING_NO_MORE_WORK;
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
            // The task has been completely applied and the execution cycle
            // is over if this is the last thread to transition to Idle
            // Finalize for the end of the cycle
            if (   (mode == MODE_RUNNING_NO_MORE_WORK) 
                && (team->N_idle_ == team->nMaxThreads_)) {
                errMsg = team->setMode_NotThreadsafe(MODE_IDLE);
                if (errMsg != "") {
                    std::string  msg = team->printState_NotThreadsafe(
                                             "threadRoutine", tId, errMsg);
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                if (team->workReceiver_) {
                    team->workReceiver_->closeTask();
                }

#ifdef VERBOSE
                team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
                team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                                  "Sent unblockWaitThread signal");
                team->logFile_.close();
#endif

                // reset team name to generic name
                team->hdr_ = "Thread Team " + std::to_string(team->id_);
                team->taskFcn_ = nullptr;

                // Make sure that this thread is ready to go Idle before
                // releasing the blocked thread
                pthread_cond_broadcast(&(team->unblockWaitThread_));
            }

            pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));
#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Activated");
            team->logFile_.close();
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
        } else if (N_Q == 0) {
            // 
            //<--------------------- TRANSITION TO WAIT --------------------->
            //
            // Should be in Running & Open

            // TODO Check for overflow 
            team->N_wait_ += 1;

            // Exclude N_terminate_ so that we also check that it is zero
            N_total =   team->N_idle_
                      + team->N_wait_
                      + team->N_comp_;
            if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Transition to Waiting");
            team->logFile_.close();
#endif
            pthread_cond_wait(&(team->transitionThread_), &(team->teamMutex_));
#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Awakened");
            team->logFile_.close();
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

            // TODO Check for overflow 
            team->N_comp_ += 1;

            N_total =   team->N_idle_
                      + team->N_wait_
                      + team->N_comp_;
            if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Transition to Computing");
            team->logFile_.close();
#endif

            work = team->queue_.front();
            team->queue_.pop();
            --N_Q;

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Dequeued work " + std::to_string(work));
            team->logFile_.close();
#endif

            // Since this emits events, only emit event after updating the mode
            // for this transition.
            //
            // No more pending work for this task (aside from the unit of work
            // just popped).  Wake up all Waiting threads so that they decide to
            // go Idle
            // 
            // NOTE: This would be handled above, but check it here so that we
            // can release waiting threads as soon as possible and therefore
            // pass these resources to thread subscribers ASAP.
            if ((mode == MODE_RUNNING_CLOSED_QUEUE) && (N_Q == 0)) {
                errMsg = team->setMode_NotThreadsafe(MODE_RUNNING_NO_MORE_WORK);
                if (errMsg != "") {
                    std::string  msg = team->printState_NotThreadsafe(
                                             "threadRoutine", tId, errMsg);
                    std::cerr << msg << std::endl;
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                pthread_cond_broadcast(&(team->transitionThread_));
            }

            // Do work!  No need to keep mutex.
            pthread_mutex_unlock(&(team->teamMutex_));
            team->taskFcn_(tId, team->hdr_, work);

            // This is where computationFinished is "emitted"
            pthread_mutex_lock(&(team->teamMutex_));

#ifdef VERBOSE
            team->logFile_.open(team->logFilename_, std::ios::out | std::ios::app);
            team->logFile_ << team->printState_NotThreadsafe("threadRoutine", tId,
                              "Finished computing");
            team->logFile_.close();
#endif

            if (team->workReceiver_) {
                team->workReceiver_->enqueue(work);
            }

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

