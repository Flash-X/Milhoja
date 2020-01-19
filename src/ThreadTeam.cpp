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
 * nMaxThreads threads in existence.  Thread teams must contain at least one
 * thread.
 * 
 * This routine initializes the state of the team in MODE_IDLE with
 *  - no threads waiting, computing, or terminating,
 *  - all nMaxThreads threads Idling, and
 *  - no pending work.
 *
 * \param  nMaxThreads The maximum permissible number of threads in the team.
 *                     Zero threads is considered to be a logical error.
 * \param  name        The name of the thread team for debug use.
 */
ThreadTeam::ThreadTeam(const unsigned int nMaxThreads,
                       const unsigned int id)
    : N_idle_actual_(0),
      N_wait_(0),
      N_comp_(0),
      queue_(),
      N_idle_intended_(nMaxThreads),
      nMaxThreads_(nMaxThreads),
      id_(id),
      hdr_(""),
      taskName_("NoTaskYet"),
      state_(nullptr),
      stateIdle_(nullptr),
      stateTerminating_(nullptr),
      stateRunOpen_(nullptr),
      stateRunClosed_(nullptr),
      stateRunNoMoreWork_(nullptr),
      taskFcn_(nullptr),
      threads_(nullptr),
      threadData_(nullptr),
      threadReceiver_(nullptr),
      workReceiver_(nullptr),
      isWaitBlocking_(false)
{
    hdr_  = "Thread Team ";
    hdr_ += std::to_string(id_);
    
    if (nMaxThreads_ == 0) {
        std::string   msg("[ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nEmpty thread teams disallowed";
        throw std::logic_error(msg);
    }

    // Initialize mutex before creating other states in case they need it 
    pthread_mutex_init(&teamMutex_, NULL);

    //***** INSTANTIATE EXTENDED FINITE STATE MACHINE STATE OBJECTS
    stateIdle_ = new ThreadTeamIdle(this); 
    if (!stateIdle_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nUnable to instantiate ";
        msg += getModeName(MODE_IDLE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateTerminating_ = new ThreadTeamTerminating(this); 
    if (!stateTerminating_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nUnable to instantiate ";
        msg += getModeName(MODE_TERMINATING);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunOpen_ = new ThreadTeamRunningOpen(this); 
    if (!stateRunOpen_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nUnable to instantiate ";
        msg += getModeName(MODE_RUNNING_OPEN_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunClosed_ = new ThreadTeamRunningClosed(this); 
    if (!stateRunClosed_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nUnable to instantiate ";
        msg += getModeName(MODE_RUNNING_CLOSED_QUEUE);
        msg += " state object";
        throw std::runtime_error(msg);
    }

    stateRunNoMoreWork_ = new ThreadTeamRunningNoMoreWork(this); 
    if (!stateRunNoMoreWork_) {
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nUnable to instantiate ";
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
    N_idle_intended_ = nMaxThreads_;
    N_wait_          = 0;
    N_comp_          = 0;
    if (!queue_.empty()) {
        pthread_mutex_unlock(&teamMutex_);
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nqueue_ is not empty ";
        throw std::runtime_error(msg);
    }
    // Setup manually since we aren't transitioning yet
    state_ = stateIdle_;

    int rc = 0;
    threads_      = new   pthread_t[nMaxThreads_];
    threadData_   = new  ThreadData[nMaxThreads_];
    for (unsigned int i=0; i<nMaxThreads_; ++i) {
        threadData_[i].tId = i;
        threadData_[i].team = this;
        // TODO: Do we need threads_ and threadData_ as arrays?
        rc = pthread_create(&threads_[i], &attr_, *threadRoutine,
                            reinterpret_cast<void*>(&(threadData_[i])));
        if (rc != 0) {
            pthread_mutex_unlock(&teamMutex_);
            std::string msg("ThreadTeam::ThreadTeam] ");
            msg += hdr_;
            msg += "\nUnable to create thread ";
            msg += std::to_string(i);
            throw std::runtime_error(msg);
        }

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] Starting thread " << i << std::endl;
#endif
    }

    // Wait until all threads have started running their routine and are Idle
    do {
        pthread_cond_wait(&threadStarted_, &teamMutex_);
    } while (N_idle_actual_ < nMaxThreads_);

    if (N_idle_actual_ != N_idle_intended_) {
        pthread_mutex_unlock(&teamMutex_);
        std::string msg("ThreadTeam::ThreadTeam] ");
        msg += hdr_;
        msg += "\nN actual idle threads ";
        msg += std::to_string(N_idle_actual_);
        msg += " not equal to N intended idle threads ";
        msg += std::to_string(N_idle_intended_);
        throw std::runtime_error(msg);
    }

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] Team initialized in state " 
                  << getModeName(state_->mode())
                  << " with "
                  << N_idle_actual_ << " threads idling\n";
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
    // Transition state and sanity check that current ThreadTeam
    // state is as expected
    std::string errMsg = setMode_NotThreadsafe(MODE_TERMINATING);
    if (errMsg != "") {
        pthread_mutex_unlock(&teamMutex_);
        std::cerr << errMsg << std::endl;
    } else if (N_idle_actual_ != nMaxThreads_) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nNumber of actual Idle threads ";
        errMsg += std::to_string(N_idle_actual_);
        errMsg += " not equal to N_max ";
        errMsg += std::to_string(nMaxThreads_);
        std::cerr << errMsg << std::endl;
    } else if (N_idle_intended_ != nMaxThreads_) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nNumber of intended Idle threads ";
        errMsg += std::to_string(N_idle_intended_);
        errMsg += " not equal to N_max ";
        errMsg += std::to_string(nMaxThreads_);
        std::cerr << errMsg << std::endl;
    } else if (N_wait_ != 0) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nNumber of Waiting threads ";
        errMsg += std::to_string(N_wait_);
        errMsg += " not zero";
        std::cerr << errMsg << std::endl;
    } else if (N_comp_ != 0) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nNumber of Computing threads ";
        errMsg += std::to_string(N_comp_);
        errMsg += " not zero";
        std::cerr << errMsg << std::endl;
    } else if (!queue_.empty()) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nPending work queue has ";
        errMsg += std::to_string(queue_.size());
        errMsg += " units of work instead of zero";
        std::cerr << errMsg << std::endl;
    } else if (threadReceiver_) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nTeam still has a Thread Subscriber";
        std::cerr << errMsg << std::endl;
    } else if (workReceiver_) {
        pthread_mutex_unlock(&teamMutex_);
        errMsg = "ThreadTeam::~ThreadTeam] ";
        errMsg += hdr_;
        errMsg += "\nTeam still has a Work Subscriber";
        std::cerr << errMsg << std::endl;
    }

    // State Transition Output
    //  - All threads are Idle and should receive this signal
    pthread_cond_broadcast(&activateThread_);

    // Block until all threads have terminated
    N_idle_intended_ = 0;
    do {
        pthread_cond_wait(&threadTerminated_, &teamMutex_);
    } while (N_idle_actual_ > N_idle_intended_);

#ifdef VERBOSE
        std::cout << "[" << hdr_ << "] " 
                  << nMaxThreads_
                  << " Threads terminated\n";
#endif

    delete [] threadData_;
    delete [] threads_;

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

#ifdef VERBOSE
    std::cout << "[" << hdr_ << "] Team destroyed\n";
#endif
}

/**
 * Obtain the name of the given mode as a string.
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
        msg += "\nInvalid state mode ";
        msg += std::to_string(mode);
        throw std::runtime_error(msg);
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
            errMsg += "\nInvalid state transition from ";
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
        errMsg += "\nUnknown ThreadTeam state mode ";
        errMsg += getModeName(nextMode);
        return errMsg;
    }

    if (!state_) {
        errMsg  = "[ThreadTeam::setMode_NotThreadsafe] ";
        errMsg += hdr_;
        errMsg += "\n";
        errMsg += getModeName(nextMode);
        errMsg += " instance is NULL";
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
 * Indicate to the thread team that it may activate the given number of Idle
 * threads so that they may help execute the current execution cycle's task.
 *
 * \warning All classes that implement this routine must confirm that activating
 * the given number of Idle threads will not exceed the maximum number of
 * threads allotted to the team.  To avoid loss of thread resources if thread
 * publishing is in use during the current cycle, these classes should use
 * N_idle_intended_ rather than N_idle_actual_.
 *
 * \param nThreads - The number of Idle threads to activate.
 */
void ThreadTeam::increaseThreadCount(const unsigned int nThreads) {
    if (!state_) {
        std::string  errMsg("ThreadTeam::increaseThreadCount] ");
        errMsg += hdr_;
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
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
        errMsg += "\nstate_ is NULL";
        throw std::runtime_error(errMsg);
    }
    state_->detachWorkReceiver();
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
 * \param  varg - a void pointer to the thread's ThreadData initializationd data
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

    bool        hasStarted = false;
    teamMode    mode = MODE_IDLE;
    while (true) {
        pthread_mutex_lock(&(team->teamMutex_));
        mode = team->state_->mode();

        if        ((mode == MODE_IDLE) && !hasStarted) {
            hasStarted = true;

            // Inform team that thread started and start in Idle
            team->N_idle_actual_ += 1;
            pthread_cond_signal(&(team->threadStarted_));
            pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));
        } else if (!hasStarted) {
            pthread_mutex_unlock(&(team->teamMutex_));
            std::string  msg("[ThreadTeam::threadRoutine] ");
            msg += team->hdr_;
            msg += "/Thread ";
            msg += std::to_string(tId);
            msg += "\nThread team not started in Idle at startup";
            throw std::logic_error(msg);
        } else if (mode == MODE_TERMINATING) {
            // Inform team that thread has terminated & terminate
            team->N_idle_actual_ -= 1;
            pthread_cond_signal(&(team->threadTerminated_));

            pthread_mutex_unlock(&(team->teamMutex_));

            pthread_exit(NULL);
            return nullptr;
        } else {
            pthread_mutex_unlock(&(team->teamMutex_));
            std::string  msg("[ThreadTeam::threadRoutine] ");
            msg += team->hdr_;
            msg += "/Thread ";
            msg += std::to_string(tId);
            msg += "\nInvalid thread control flow";
            throw std::logic_error(msg);
        }
        pthread_mutex_unlock(&(team->teamMutex_));
    }

    std::string  msg("[ThreadTeam::threadRoutine] ");
    msg += team->hdr_;
    msg += "/Thread ";
    msg += std::to_string(tId);
    msg += "\nInvalid thread control flow";
    throw std::logic_error(msg);
}

