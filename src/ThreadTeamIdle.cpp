#include "ThreadTeamIdle.h"

/**
 * Instantiate a ThreadTeamIdle object for internal use by a ThreadTeam object
 * as part of the State design pattern.  This gives the concrete state object a
 * pointer to the ThreadTeam object whose private data members it will directly
 * adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamIdle::ThreadTeamIdle(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamIdle::ThreadTeamIdle] ");
        msg += team_->hdr_;
        msg += "\n\tGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamIdle::mode(void) const {
    return ThreadTeam::MODE_IDLE;
}

/**
 * Confirm that the state of the EFSM is valid for the Idle mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamIdle::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if (team_->N_idle_ != team_->nMaxThreads_) {
        errMsg  = "N_idle != N threads in team";
    } else if (team_->N_wait_ != 0) {
        errMsg = "N_wait not zero";
    } else if (team_->N_comp_ != 0) {
        errMsg = "N_comp not zero";
    } else if (team_->N_terminate_ != 0) {
        errMsg = "N_terminate not zero";
    } else if (!team_->queue_.empty()) {
        errMsg = "Pending work queue not empty";
   }

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamIdle::startTask_NotThreadsafe(TASK_FCN* fcn,
                                                    const unsigned int nThreads,
                                                    const std::string& teamName, 
                                                    const std::string& taskName) {
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[" << team_->hdr_ << "] Assigned team name "
                    << teamName << std::endl;
    team_->logFile_ << "[" << teamName << "] Starting task "
                    << taskName
                    << " with "
                    << std::to_string(nThreads)
                    << " initial threads\n";
    team_->logFile_.close();
#endif

    team_->hdr_ = teamName;
    team_->taskFcn_ = fcn;

    team_->N_to_activate_ += nThreads;
    team_->setMode_NotThreadsafe(ThreadTeam::MODE_RUNNING_OPEN_QUEUE);
    for (unsigned int i=0; i<nThreads; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Pass the threads along to a thread subscriber if the object has one.
 * Otherwise, it does nothing.  This behavior is motivated by the case that a
 * Thread Subscriber has gone Idle before its Thread Publisher has finished its
 * task.  Note that all subsequent teams that are Idle and that receive this
 * signal will ignore it as well since adding threads to a team with no task is
 * nonsensical.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamIdle::increaseThreadCount_NotThreadsafe(
                                    const unsigned int nThreads) {
    // No need to alter N_to_activate_ as we forward the thread activation on
    if (team_->threadReceiver_) {
        team_->threadReceiver_->increaseThreadCount(nThreads);
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * To simplify the design, we do not allow for adding work in Idle mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string  ThreadTeamIdle::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Adding work in Idle state not allowed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Calling this is a logic error as we cannot close the queue if no task is
 * running.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamIdle::closeTask_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeTask", 0,
                  "Cannot close the queue when no task is being executed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Calling wait on a ThreadTeam that is Idle seems like a logic error.  However,
 * it could be that a team finishes its task and transition to Idle before a
 * calling thread got a chance to call wait().  Therefore, this method is a
 * no-op so that it won't block.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string    ThreadTeamIdle::wait_NotThreadsafe(void) {
#ifdef VERBOSE
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[Client Thread] Called no-op wait (Idle)\n";
    team_->logFile_.close();
#endif

    return "";
}

