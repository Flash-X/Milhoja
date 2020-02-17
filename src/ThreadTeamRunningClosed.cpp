#include "ThreadTeamRunningClosed.h"

/**
 * Instantiate a ThreadTeamRunningClosed object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamRunningClosed::ThreadTeamRunningClosed(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningClosed::ThreadTeamRunningClosed] ");
        msg += team_->hdr_;
        msg += "\n\tGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Destroy the concrete state object.
 */
ThreadTeamRunningClosed::~ThreadTeamRunningClosed(void) { }

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamRunningClosed::mode(void) const {
    return ThreadTeam::MODE_RUNNING_CLOSED_QUEUE;
}

/**
 * 
 */
std::string ThreadTeamRunningClosed::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if (team_->N_terminate_ != 0) {
        errMsg = "N_terminate not zero";
    }

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                        const std::string& teamName, 
                                        const std::string& taskName) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "startTask", 0, "Cannot start a task when one is already running");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::increaseThreadCount(const unsigned int nThreads) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "increaseThreadCount", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (nThreads == 0) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "increaseThreadCount", 0, "No sense in increasing by zero threads");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (nThreads > (team_->N_idle_ - team_->N_to_activate_)) {
        msg  = "nThreads (";
        msg += std::to_string(nThreads);
        msg += ") exceeds the number of threads available for activation";
        std::string  errMsg = team_->printState_NotThreadsafe(
            "increaseThreadCount", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }
 
    // Don't activate all threads if we are asked to activate more threads than
    // this is remaining work
    // NOTE: This also handles the case that the queue is empty
    unsigned int   nWork = team_->queue_.size();
    unsigned int   nThreadsMin = 0;
    if (nThreads > nWork) {
        nThreadsMin = nWork;
    } else {
        nThreadsMin = nThreads;
    }

    team_->N_to_activate_ += nThreadsMin;
    for (unsigned int i=0; i<nThreadsMin; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    if ((nThreads > nWork) && (team_->threadReceiver_)) {
        team_->threadReceiver_->increaseThreadCount(nThreads - nWork);
    }

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::enqueue(const int work) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "enqueue", 0, "Cannot enqueue work if queue is closed");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::closeTask() {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "closeTask", 0, "The task is already closed");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::wait(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "wait", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (team_->isWaitBlocking_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "wait", 0, "A thread has already called wait");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    // Block until team transitions to Idle
    team_->isWaitBlocking_ = true;
    pthread_cond_wait(&(team_->unblockWaitThread_), &(team_->teamMutex_));
    team_->isWaitBlocking_ = false;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::attachThreadReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "attachThreadReceiver", 0, "Subscribers can only be attached in Idle");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::detachThreadReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "detachThreadReceiver", 0, "Subscribers can only be detached in Idle");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}
    
/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::attachWorkReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "attachWorkReceiver", 0, "Subscribers can only be attached in Idle");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningClosed::detachWorkReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "detachWorkReceiver", 0, "Subscribers can only be detached in Idle");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

