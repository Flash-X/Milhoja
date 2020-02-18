#include "ThreadTeamRunningOpen.h"

/**
 * Instantiate a ThreadTeamRunningOpen object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamRunningOpen::ThreadTeamRunningOpen(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningOpen::ThreadTeamRunningOpen] ");
        msg += team_->hdr_;
        msg += "\n\tGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Destroy the concrete state object.
 */
ThreadTeamRunningOpen::~ThreadTeamRunningOpen(void) { }

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamRunningOpen::mode(void) const {
    return ThreadTeam::MODE_RUNNING_OPEN_QUEUE;
}

/**
 * 
 */
std::string ThreadTeamRunningOpen::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if (team_->N_terminate_ != 0) {
        errMsg = "N_terminate not zero";
    }

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 */
std::string ThreadTeamRunningOpen::startTask_NotThreadsafe(TASK_FCN* fcn,
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
std::string ThreadTeamRunningOpen::increaseThreadCount_NotThreadsafe(
                                            const unsigned int nThreads) {
    team_->N_to_activate_ += nThreads;
    for (unsigned int i=0; i<nThreads; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
std::string ThreadTeamRunningOpen::enqueue_NotThreadsafe(const int work) {
    team_->queue_.push(work);
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[" << team_->hdr_ << "] Enqueued work " << work << std::endl;
    team_->logFile_.close();
#endif

    // Wake a waiting thread (if there is one) so that it can start
    // applying the task to the new work
    pthread_cond_signal(&(team_->transitionThread_));

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
std::string ThreadTeamRunningOpen::closeTask_NotThreadsafe(void) {
    if (team_->queue_.empty()) {
        team_->setMode_NotThreadsafe(ThreadTeam::MODE_RUNNING_NO_MORE_WORK);
        pthread_cond_broadcast(&(team_->transitionThread_));
    } else {
        team_->setMode_NotThreadsafe(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE);
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
std::string  ThreadTeamRunningOpen::wait_NotThreadsafe(void) {
    team_->isWaitBlocking_ = true;
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Waiting on team (Run & Open)\n";
    team_->logFile_.close();
#endif
    pthread_cond_wait(&(team_->unblockWaitThread_), &(team_->teamMutex_));
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Received unblockWaitSignal (Run & Open)\n";
    team_->logFile_.close();
#endif
    team_->isWaitBlocking_ = false;
        
    return "";
}

