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
 * Confirm that the state of the EFSM is valid for the Running & Queue Closed mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::isStateValid_NotThreadSafe(void) const {
    if (team_->N_terminate_ != 0) {
        return "N_terminate not zero";
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not start a task if one is still running.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::startTask_NotThreadsafe(TASK_FCN* fcn,
                                                             const unsigned int nThreads,
                                                             const std::string& teamName, 
                                                             const std::string& taskName) {
    return team_->printState_NotThreadsafe("startTask", 0,
                  "Cannot start a task when one is already running");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::increaseThreadCount_NotThreadsafe(
                                                const unsigned int nThreads) {
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

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * No more work can be added once the queue has been closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue work if queue is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Can' close a task that is already closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::closeTask_NotThreadsafe(void) {
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
std::string  ThreadTeamRunningClosed::wait_NotThreadsafe(void) {
    // Block until team transitions to Idle
    team_->isWaitBlocking_ = true;

#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Waiting on team (Run & Closed)\n";
    team_->logFile_.close();
#endif

    pthread_cond_wait(&(team_->unblockWaitThread_), &(team_->teamMutex_));

#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Received unblockWaitThread (Run & Closed)\n";
    team_->logFile_.close();
#endif

    team_->isWaitBlocking_ = false;

    return "";
}

