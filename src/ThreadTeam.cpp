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
      taskName_("No Task Yet"),
      taskFcn_(nullptr),
      threadReceiver_(nullptr),
      workReceiver_(nullptr),
      isWaitBlocking_(false)
{
    hdr_  = "Thread Team ";
    hdr_ += std::to_string(id_);
    
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
    N_idle_      = 0;
    N_wait_      = 0;
    N_comp_      = 0;
    N_terminate_ = 0;
    if (!queue_.empty()) {
        std::string  msg = printState_NotThreadsafe(
            "ThreadTeam", 0, "Work queue is not empty");
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
            std::string  msg = printState_NotThreadsafe(
                "ThreadTeam", i, "Unable to create thread");
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error(msg);
        }

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] Starting thread " << i << std::endl;
#endif
    }

    // Wait until all threads have started running their routine and are Idle
    do {
        // TODO: Timeout on this?
        pthread_cond_wait(&threadStarted_, &teamMutex_);
    } while (N_idle_ < nMaxThreads_);
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
        std::string  errMsg = printState_NotThreadsafe(
            "ThreadTeam", 0, msg);
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error(errMsg);
    }

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] Team initialized in state " 
                  << getModeName(state_->mode())
                  << " with "
                  << N_idle_ << " threads idling\n";
#endif

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Destroy the thread team.
 * 
 * \warning: This should only be run if the team is in MODE_IDLE.
 */
ThreadTeam::~ThreadTeam(void) {
    pthread_mutex_lock(&teamMutex_);

    // TODO: How to handle errors discovered in destructor?
    std::string msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, msg);
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    } else if (threadReceiver_) {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, "Team still has thread subscriber");
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    } else if (workReceiver_) {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, "Team still has work subscriber");
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    }

    // Transition state and sanity check that current ThreadTeam
    // state is as expected
    msg = setMode_NotThreadsafe(MODE_TERMINATING);
    if (msg != "") {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, msg);
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    }

    // State Transition Output
    //  - All threads are Idle and should receive this signal
    //  - Call this only after changing mode to Terminating
    N_to_activate_ = nMaxThreads_;
    pthread_cond_broadcast(&activateThread_);

    // Block until all threads have terminated
    do {
        // TODO: Timeout on this?
        pthread_cond_wait(&threadTerminated_, &teamMutex_);
    } while (N_idle_ > 0);

    unsigned int N_total =   N_idle_ + N_wait_ 
                           + N_comp_ + N_terminate_;
    if (N_total != nMaxThreads_) {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, "Inconsistent thread counts");
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    } else if (N_terminate_ != nMaxThreads_) {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, "N_terminate != N_max");
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    } else if (N_to_activate_ != 0) {
        std::string  errMsg = printState_NotThreadsafe(
            "~ThreadTeam", 0, "N_to_activate != 0");
        std::cerr << errMsg << std::endl;
        pthread_mutex_unlock(&teamMutex_);
    }

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] " 
                  << nMaxThreads_
                  << " Threads terminated\n";
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
    delete stateIdle_;
    delete stateTerminating_;
    delete stateRunOpen_;
    delete stateRunClosed_;
    delete stateRunNoMoreWork_;
    stateIdle_ = nullptr;
    stateTerminating_ = nullptr;
    stateRunOpen_ = nullptr;
    stateRunClosed_ = nullptr;
    stateRunNoMoreWork_ = nullptr;

#ifdef VERBOSE
    std::cout << "[" << hdr_ << "] Team destroyed\n";
#endif
}

/**
 * Obtain the name of the given mode as a string.  This mode does not access
 * resources and therefore does not need to be called only after acquiring the
 * team's mutex.
 *
 * \param   mode - The enum value of the mode.
 * \return  The name.
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
 *
 * Set the mode of the EFSM to the given mode.  This internal method is usually
 * called as part of a state transition.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \param    nextMode - The next mode as an enum value.
 * \return   An empty string if the mode was set successfully, an error
 *           statement otherwise.
 */
std::string ThreadTeam::setMode_NotThreadsafe(const teamMode nextMode) {
    std::string    errMsg("");

    teamMode  currentMode = state_->mode();

    switch(nextMode) {
    case MODE_IDLE:
        state_ = stateIdle_;
        break;
    case MODE_TERMINATING:
        // This transition is not initiated by one of the State objects, but
        // rather by clients calling the Destructor.  Therefore, we have to
        // error check the Mode transitions here.
        if (currentMode != MODE_IDLE) {
            errMsg  = "[ThreadTeam::setMode_NotThreadsafe] ";
            errMsg += hdr_;
            errMsg += "\n\tInvalid state transition from ";
            errMsg += getModeName(currentMode);
            errMsg += " to ";
            errMsg += getModeName(nextMode);
            return errMsg;
        }
        state_ = stateTerminating_;
        break;
    default:
        errMsg  = "[ThreadTeam::setMode_NotThreadsafe] ";
        errMsg += hdr_;
        errMsg += "\n\tUnknown ThreadTeam state mode ";
        errMsg += getModeName(nextMode);
        return errMsg;
    }

    std::string msg("");
    if (!state_) {
        msg  = getModeName(nextMode);
        msg += " instance is NULL";
        std::string  errMsg = printState_NotThreadsafe(
            "setMode_NotThreadsafe", 0, msg);
        return errMsg;
    } 
    msg = state_->isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = printState_NotThreadsafe(
            "setMode_NotThreadsafe", 0, msg);
        return errMsg;
    }

#ifdef VERBOSE
    std::cout << "[" << hdr_ << "] Transitioned from "
              << getModeName(currentMode)
              << " to "
              << getModeName(nextMode) << std::endl;
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
    return nMaxThreads_;
}

/**
 *
 */
ThreadTeam::teamMode ThreadTeam::mode(void) const {
    if (!state_) {
        std::string  errMsg("ThreadTeam::mode] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->mode();
}

/**
 * Indicate to the thread team that it may activate the given number of Idle
 * threads so that they may help execute the current execution cycle's task.
 *
 * \warning All classes that implement this routine must confirm that activating
 * the given number of Idle threads will not lead to the case that the
 * total number of activated threads exceed the maximum number of threads
 * allotted to the team.  To avoid loss of thread resources if thread publishing
 * is in use during the current cycle, these classes should assume that the
 * number of Idle threads available for activation is N_idle_ - N_to_activate_
 * and *not* the actual number of threads that are currently Idle (i.e. N_idle).
 * These implementations must also update the value of N_to_activate_
 * appropriately.
 *
 * \param nThreads - The number of Idle threads to activate.
 */
void ThreadTeam::increaseThreadCount(const unsigned int nThreads) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::increaseThreadCount] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->increaseThreadCount(nThreads);
}

/**
 * Start an execution cycle that will apply the given task to all tiles
 * subsequently given to the team using the enqueue() method.
 * 
 * Note that all tasks must conform to the same minimal interface.  This
 * constraint avoids exposing computation-specific information to the Thread
 * Teams so that these are decoupled from the content of the work being done.
 * Client code must make all task-/computation-specific data available to the
 * task function through other means.
 *
 * An execution cycle can begin with zero threads with the understanding that
 * the team will be setup as a thread subscriber so that threads can be
 * activated at a later time by the thread publisher.
 *
 * \warning All classes that implement this routine must confirm that activating
 * the given number of Idle threads will not lead to the case that the
 * total number of activated threads exceed the maximum number of threads
 * allotted to the team.  To avoid loss of thread resources if thread publishing
 * is in use during the current cycle, these classes should assume that the
 * number of Idle threads available for activation is N_idle_ - N_to_activate_
 * and *not* the actual number of threads that are currently Idle (i.e. N_idle).
 * These implementations must also update the value of N_to_activate_
 * appropriately.
 *
 * \param    fcn - a function pointer to the task to execute.
 * \param    nThreads - the number of Idle threads to immediately activate.
 * \param    teamName - a name to assign to the team that will be used for
 *                      logging the team during this execution cycle.
 * \param    taskName - a name to assign to the task that will be used for
 *                      logging the team during this execution cycle.
 */
void ThreadTeam::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                           const std::string& teamName, 
                           const std::string& taskName) {
    // TODO: Is it OK that we use state_ without acquiring the mutex?  state_
    // should only be altered by calling setMode_NotThreadsafe, which can be
    // called by the object pointed to by state_.  Better to acquire the mutex
    // here and pass this through as a parameter so that a called method can
    // release the mutex on error?  This seems ugly but safer.
    if (!state_) {
        std::string  errMsg("ThreadTeam::startTask] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->startTask(fcn, nThreads, teamName, taskName);
}

/**
 * Indicate to the thread team that no more units of work will be given to the
 * team during the present execution cycle.
 */
void ThreadTeam::closeTask(void) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::closeTask] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->closeTask();
}

/**
 * Give the thread team a unit of work on which it should apply the current
 * execution cycle's task.
 *
 * \todo    Need to figure out how to manage more than one unit of work.
 *          For a CPU-heavy task, it should receive tiles; a GPU-heavy
 *          task, a data packet of blocks.
 *
 * \param   work - the unit of work.
 */
void ThreadTeam::enqueue(const int work) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::enqueue] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->enqueue(work);
}

/**
 * Block a single calling thread until the team finishes executing its current
 * execution cycle.
 *
 * \warning Classes derived from ThreadTeamState need to ensure that only one
 *          single thread can be blocked by calling wait() at a time.
 */
void ThreadTeam::wait(void) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::wait] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->wait();
}

/**
 * Attach a thread team as a thread subscriber.  Therefore, this converts the
 * calling object into a thread publisher.
 *
 * \warning Classes derived from ThreadTeamState need to ensure that only one
 *          thread team can be attached as a thread subscriber at a time.  Also,
 *          they must confirm that the pointer is non-null, and that the team
 *          itself is not being set to publish threads to itself.
 *
 * \param  receiver - the team to which thread transitions to Idle shall by
 *                    published.
 */
void ThreadTeam::attachThreadReceiver(ThreadTeam* receiver) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::attachThreadReceiver] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->attachThreadReceiver(receiver);
}

/**
 * Detach the thread subscriber so that the calling object is no longer a thread
 * publisher.
 *
 * \warning Classes derived from ThreadTeamState should consider it a logical
 * error if this routine is called when the calling object has no thread team
 * attached as a subscriber.
 */
void ThreadTeam::detachThreadReceiver(void) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::detachThreadReceiver] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->detachThreadReceiver();
}

/**
 * Register a thread team as a work subscriber.  Therefore, this converts the
 * calling object into a work publisher.
 *
 * \warning Classes derived from ThreadTeamState need to ensure that only one
 *          thread team can be attached as a work subscriber at a time.  Also,
 *          they must confirm that the pointer is non-null, and that the team
 *          itself is not being set to publish work to itself.
 *
 * \param  receiver - the team to which units of work shall be published.
 */
void ThreadTeam::attachWorkReceiver(ThreadTeam* receiver) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::attachWorkReceiver] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->attachWorkReceiver(receiver);
}

/**
 * Detach the work subscriber so that the calling object is no longer a work
 * publisher.
 *
 * \warning Classes derived from ThreadTeamState should consider it a logical
 * error if this routine is called when the calling object has no thread team
 * attached as a subscriber.
 */
void ThreadTeam::detachWorkReceiver(void) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::detachWorkReceiver] ");
        errMsg += hdr_;
        errMsg += "\n\tstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->detachWorkReceiver();
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
    std::string  state("[ThreadTeam::");
    state += method;
    state += "] ";
    state += hdr_;
    state += "/Thread ";
    state += std::to_string(tId);
    state += "\n\tError - ";
    state += msg;
    state += "\n\tThread team state snapshot";
    state += "\n\t--------------------------------------------------------";
    state += "\n\tMode\t\t\t\t";
    state += getModeName(state_->mode());
    state += "\n\tN threads in team\t\t";
    state += std::to_string(nMaxThreads_);
    state += "\n\tN threads actually Idle\t\t";
    state += std::to_string(N_idle_);
    state += "\n\tN threads Waiting\t\t";
    state += std::to_string(N_wait_);
    state += "\n\tN threads Computing\t\t";
    state += std::to_string(N_comp_);
    state += "\n\tN threads Terminating\t\t";
    state += std::to_string(N_terminate_);
    state += "\n\tN units of work in queue\t";
    state += std::to_string(queue_.size());
    state += "\n\tN threads pending activation\t";
    state += std::to_string(N_to_activate_);
    state += "\n";

    return state;
}

/**
 * This is the routine that each thread in the team executes as it moves
 * through the thread states Idle, Waiting, Computing, and Terminating in accord
 * with the team's mode, the queue contents and the state of the other threads
 * in the team.
 *
 * A thread is
 *   - Idle if it is waiting on the activateThread_ signal
 *   - Terminating if it has determined that the team has mode Terminating
 *   - Waiting if it is not Idle or Terminating and waiting on transitionThread_
 *   - Computing if it found work at the last transition and is currently
 *     executing the task on that work.
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
    //   the thread waits on a signal and therefore relinquishes the mutex.
    //   The transition has finished.
    //   Note that for a computing thread "wait" means block while it applies
    //   the task to its unit of work.  The termination of the work is the
    //   "signal" that the thread sends to itself to transition.
    // - the thread then loops around to repeat the process.
    // TODO: When is the output executed?
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

    unsigned int N_total      = 0;
    teamMode     mode         = MODE_IDLE;
    bool         hasStarted   = false;
    bool         isQueueEmpty = false;
    while (true) {
        mode = team->state_->mode();
        isQueueEmpty = team->queue_.empty();

        if (mode == MODE_IDLE) {
            team->N_idle_ += 1;

            if (hasStarted) {
                N_total =   team->N_idle_ + team->N_wait_
                          + team->N_comp_ + team->N_terminate_;
                if (N_total != team->nMaxThreads_) {
                    std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, "Inconsistent thread counts");
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::logic_error(msg);
                }
            } else {
                // If starting thread team, we cannot yet check that
                // N_total law is satisfied - the constructor confirms this
                hasStarted = true;
                pthread_cond_signal(&(team->threadStarted_));
            }

            // It is possible that a call to increaseThreadCount could result
            // in the emission of activateThread event before a team goes Idle
            // and the event received after the team goes idle.  If this occurs,
            // forward the thread count increase and stay Idle.
            do {
                pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));

                mode = team->state_->mode();
                isQueueEmpty = team->queue_.empty();
                if (mode == MODE_IDLE) {
                    if (team->threadReceiver_) {
                        team->threadReceiver_->increaseThreadCount(1);
                    }

                    // Confirm expected thread counts
                    if        (team->N_idle_ != team->nMaxThreads_) {
                        std::string  msg = team->printState_NotThreadsafe(
                            "threadRoutine", tId, "N_idle != N_max");
                        pthread_mutex_unlock(&(team->teamMutex_));
                        throw std::runtime_error(msg);
                    } else if (team->N_wait_ != 0) {
                        std::string  msg = team->printState_NotThreadsafe(
                            "threadRoutine", tId, "N_wait is not zero");
                        pthread_mutex_unlock(&(team->teamMutex_));
                        throw std::runtime_error(msg);
                    } else if (team->N_comp_ != 0) {
                        std::string  msg = team->printState_NotThreadsafe(
                            "threadRoutine", tId, "N_comp is not zero");
                        pthread_mutex_unlock(&(team->teamMutex_));
                        throw std::runtime_error(msg);
                    } else if (team->N_terminate_ != 0) {
                        std::string  msg = team->printState_NotThreadsafe(
                            "threadRoutine", tId, "N_terminate is not zero");
                        pthread_mutex_unlock(&(team->teamMutex_));
                        throw std::runtime_error(msg);
                    } else if (!isQueueEmpty) {
                        std::string  msg = team->printState_NotThreadsafe(
                            "threadRoutine", tId, "Work queue is not empty");
                        pthread_mutex_unlock(&(team->teamMutex_));
                        throw std::runtime_error(msg);
                    }
                }

                if (team->N_to_activate_ <= 0) {
                    std::string  msg = team->printState_NotThreadsafe(
                        "threadRoutine", tId, 
                        "Thread activated with N_to_activate_ zero");
                    pthread_mutex_unlock(&(team->teamMutex_));
                    throw std::runtime_error(msg);
                }
                team->N_to_activate_ -= 1;
            } while (mode == MODE_IDLE);

            if (team->N_idle_ <= 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_idle unexpectedly zero");
                pthread_mutex_unlock(&(team->teamMutex_));
                throw std::runtime_error(msg);
            }
            team->N_idle_ -= 1;
        } else if (mode == MODE_TERMINATING) {
            team->N_terminate_ += 1;
            N_total =   team->N_idle_ + team->N_wait_
                      + team->N_comp_ + team->N_terminate_;

            // Report errors rather than throw errors since
            // this runs when the team object is in destructor
            if        (team->N_wait_ != 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_wait not zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            } else if (team->N_comp_ != 0) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "N_comp not zero");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            } else if (N_total != team->nMaxThreads_) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Inconsistent thread counts");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            } else if (!isQueueEmpty) {
                std::string  msg = team->printState_NotThreadsafe(
                    "threadRoutine", tId, "Work queue not empty");
                std::cerr << msg << std::endl;
                pthread_mutex_unlock(&(team->teamMutex_));
            }

            // Inform team that thread has terminated & terminate
            pthread_cond_signal(&(team->threadTerminated_));
            pthread_mutex_unlock(&(team->teamMutex_));
            pthread_exit(NULL);
            return nullptr;
        } else {
            std::string  msg = team->printState_NotThreadsafe(
                "threadRoutine", tId, "Invalid thread control flow");
            pthread_mutex_unlock(&(team->teamMutex_));
            throw std::logic_error(msg);
        }
    }
    pthread_mutex_unlock(&(team->teamMutex_));

    std::string  msg = team->printState_NotThreadsafe(
        "threadRoutine", tId, "Thread unexpectedly reached end of routine");
    throw std::logic_error(msg);
}

