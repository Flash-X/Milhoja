#include "ThreadTeamRunningClosed.h"

/**
 * Instantiate a ThreadTeamRunningClosed object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
template<class T>
ThreadTeamRunningClosed<T>::ThreadTeamRunningClosed(T* team)
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
template<class T>
std::string ThreadTeamRunningClosed<T>::isStateValid_NotThreadSafe(void) const {
    if        (team_->N_terminate_ != 0) {
        return "N_terminate not zero";
    } else if (team_->queue_.empty()) {
        return "The pending work queue should not be empty";
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
template<class T>
std::string ThreadTeamRunningClosed<T>::startTask_NotThreadsafe(TASK_FCN* fcn,
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
template<class T>
std::string ThreadTeamRunningClosed<T>::increaseThreadCount_NotThreadsafe(
                                                const unsigned int nThreads) {
    // Don't activate all threads if we are asked to activate more threads than
    // there is remaining work
    unsigned int   N_Q = team_->queue_.size();
    unsigned int   nThreadsMin = nThreads;
    if (nThreads > N_Q) {
        nThreadsMin = N_Q;
    } 

    team_->N_to_activate_ += nThreadsMin;
    for (unsigned int i=0; i<nThreadsMin; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    if ((nThreads > N_Q) && (team_->threadReceiver_)) {
        team_->threadReceiver_->increaseThreadCount(nThreads - N_Q);
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
template<class T>
std::string ThreadTeamRunningClosed<T>::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue work if queue is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Can't close a task that is already closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<class T>
std::string ThreadTeamRunningClosed<T>::closeTask_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeTask", 0,
                  "The task is already closed");
}

