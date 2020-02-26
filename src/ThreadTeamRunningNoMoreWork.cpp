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
 * Confirm that the state of the EFSM is valid for the Running/No More Work mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningNoMoreWork::isStateValid_NotThreadSafe(void) const {
    if (team_->N_terminate_ != 0) {
        return "N_terminate not zero";
    } else if (team_->N_idle_ == team_->nMaxThreads_) {
        return "At least one thread should be active";
    } else if (!(team_->queue_.empty())) {
        return "Pending work queue not empty";
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Cannot start a task when one is still running.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
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
 * Forward threads onto thread subscriber.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningNoMoreWork::increaseThreadCount_NotThreadsafe(
                                                    const unsigned int nThreads) {
    if (team_->threadReceiver_) {
        team_->threadReceiver_->increaseThreadCount(nThreads);
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * The queue is closed.  No work can be added.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningNoMoreWork::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue work if queue is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * The task is already closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningNoMoreWork::closeTask_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeTask", 0,
                  "The task is already closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
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

