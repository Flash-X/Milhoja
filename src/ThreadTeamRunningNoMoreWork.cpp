#include "ThreadTeamRunningNoMoreWork.h"

/**
 * Instantiate a ThreadTeamRunningNoMoreWork object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamRunningNoMoreWork::ThreadTeamRunningNoMoreWork(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningNoMoreWork::ThreadTeamRunningNoMoreWork] ");
        msg += team_->hdr_;
        msg += "\n\tGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Destroy the concrete state object.
 */
ThreadTeamRunningNoMoreWork::~ThreadTeamRunningNoMoreWork(void) { }

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamRunningNoMoreWork::mode(void) const {
    return ThreadTeam::MODE_RUNNING_NO_MORE_WORK;
}

/**
 * 
 */
std::string ThreadTeamRunningNoMoreWork::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if (team_->N_terminate_ != 0) {
        errMsg = "N_terminate not zero";
    } else if (!(team_->queue_.empty())) {
        errMsg = "Pending work queue not empty";
    }

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
std::string ThreadTeamRunningNoMoreWork::startTask_NotThreadsafe(TASK_FCN* fcn,
                                                                 const unsigned int nThreads,
                                                                 const std::string& teamName, 
                                                                 const std::string& taskName) {
    return team_->printState_NotThreadsafe("startTask", 0,
                  "Cannot start a task when one is already running");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningNoMoreWork::increaseThreadCount(const unsigned int nThreads) {
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
 
    if (team_->threadReceiver_) {
        team_->threadReceiver_->increaseThreadCount(nThreads);
    }

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
std::string ThreadTeamRunningNoMoreWork::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue work if queue is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningNoMoreWork::closeTask() {
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
std::string   ThreadTeamRunningNoMoreWork::wait_NotThreadsafe(void) {
    team_->isWaitBlocking_ = true;
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Waiting on team (No More Work)\n";
    team_->logFile_.close();
#endif
    pthread_cond_wait(&(team_->unblockWaitThread_), &(team_->teamMutex_));
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Received unblockWaitThread (No More Work)\n";
    team_->logFile_.close();
#endif
    team_->isWaitBlocking_ = false;

    return "";
}

