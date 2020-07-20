#include "ThreadTeamRunningNoMoreWork.h"

namespace orchestration {

/**
 * Instantiate a ThreadTeamRunningNoMoreWork object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
template<typename DT, class T>
ThreadTeamRunningNoMoreWork<DT,T>::ThreadTeamRunningNoMoreWork(T* team)
    : ThreadTeamState<DT,T>(),
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
template<typename DT, class T>
std::string ThreadTeamRunningNoMoreWork<DT,T>::isStateValid_NotThreadSafe(void) const {
    if (team_->N_terminate_ != 0) {
        return "N_terminate not zero";
    } else if (team_->N_idle_ == team_->nMaxThreads_) {
        return "At least one thread should be active";
    } else if (!(team_->queue_.empty())) {
        return "Data item queue not empty";
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Cannot start a cycle when one is still running.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename DT, class T>
std::string ThreadTeamRunningNoMoreWork<DT,T>::startCycle_NotThreadsafe(const RuntimeAction& action,
                                                                        const std::string& teamName) {
    return team_->printState_NotThreadsafe("startCycle", 0,
                  "Cannot start a cycle when one is already running");
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
template<typename DT, class T>
std::string ThreadTeamRunningNoMoreWork<DT,T>::increaseThreadCount_NotThreadsafe(
                                                    const unsigned int nThreads) {
    if (team_->threadReceiver_) {
        team_->threadReceiver_->increaseThreadCount(nThreads);
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * The cycle is closed.  No data items can be added.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename DT, class T>
std::string ThreadTeamRunningNoMoreWork<DT,T>::enqueue_NotThreadsafe(std::shared_ptr<DT>&& dataItem) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot enqueue data items if cycle is closed");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * The data item queue is already closed.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename DT, class T>
std::string ThreadTeamRunningNoMoreWork<DT,T>::closeQueue_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeQueue", 0,
                  "The data item queue is already closed");
}
}
