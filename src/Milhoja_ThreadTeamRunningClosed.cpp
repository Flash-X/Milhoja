#include "Milhoja_ThreadTeamRunningClosed.h"

#include <stdexcept>

#include "Milhoja_DataItem.h"
#include "Milhoja_ThreadTeam.h"

namespace milhoja {

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
 * Do not start an execution cycle if one is still running.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::startCycle_NotThreadsafe(const RuntimeAction& action,
                                                              const std::string& teamName) {
    return team_->printState_NotThreadsafe("startCycle", 0,
                  "Cannot start a cycle when one is already running");
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
 * No more data items can be added once the execution cycle has been closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::enqueue_NotThreadsafe(std::shared_ptr<DataItem>&& dataItem) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue data item if cycle is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Can't close data item queue if it has already been closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
std::string ThreadTeamRunningClosed::closeQueue_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeQueue", 0,
                  "Data item queue is already closed");
}
}
