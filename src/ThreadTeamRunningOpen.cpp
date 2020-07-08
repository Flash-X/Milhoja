#include "ThreadTeamRunningOpen.h"
namespace orchestration {

/**
 * Instantiate a ThreadTeamRunningOpen object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
template<typename W, class T>
ThreadTeamRunningOpen<W,T>::ThreadTeamRunningOpen(T* team)
    : ThreadTeamState<W,T>(),
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
 * Confirm that the state of the EFSM is valid for the Running & Queue Open  mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamRunningOpen<W,T>::isStateValid_NotThreadSafe(void) const {
    if (team_->N_terminate_ != 0) {
        return "N_terminate not zero";
    }

    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not allow for starting a new task if one is still on-going.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamRunningOpen<W,T>::startTask_NotThreadsafe(const RuntimeAction& action,
                                                                const std::string& teamName) {
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
template<typename W, class T>
std::string ThreadTeamRunningOpen<W,T>::increaseThreadCount_NotThreadsafe(
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
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamRunningOpen<W,T>::enqueue_NotThreadsafe(W& work, const bool move) {
    if (move) {
        // Force move to new Tile object so that the object pushed to queue has
        // ownership of tile resources
        team_->queue_.push(std::move(work));
    } else {
        // Copy the tile resources and push copy on queue.  The copy gets its
        // own tile resources.
        team_->queue_.push(work);
    }

#ifdef DEBUG_RUNTIME
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[" << team_->hdr_ << "] Enqueued work\n";
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
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamRunningOpen<W,T>::closeTask_NotThreadsafe(void) {
    std::string    errMsg("");

    bool isQueueEmpty = team_->queue_.empty();
    if        (    isQueueEmpty
               && (team_->N_idle_ == team_->nMaxThreads_) ) {
        // No more work can be added and there are no threads that are active
        //   => no need to transition threads.
        // If N_to_activate_ > 0, then activated threads will transition back
        // to Idle based on the new Mode
        errMsg = team_->setMode_NotThreadsafe(ThreadTeamMode::IDLE);
        if (errMsg != "") {
            return errMsg;
        }

        if (team_->workReceiver_) {
            team_->workReceiver_->closeTask();
        }
    } else if (isQueueEmpty) {
        // No more work, but we have threads that need to transition to Idle
        // - Awaken Waiting threads so that they find no work and transition
        // - Computing threads will find no work eventually and transition
        errMsg = team_->setMode_NotThreadsafe(ThreadTeamMode::RUNNING_NO_MORE_WORK);
        if (errMsg != "") {
            return errMsg;
        }
        pthread_cond_broadcast(&(team_->transitionThread_));
    } else {
        // We could add an optimization here.  If there are N waiting threads
        // and M<N units of pending work, then we could transition N-M waiting
        // threads to Idle to free up resources as quickly as possible.  To do
        // this properly, we would need to keep track of the actual number of
        // waiting threads as well as the number of transitionThread events
        // emitted but not yet received.  Not presently worth the effort or
        // increased complexity.
        errMsg = team_->setMode_NotThreadsafe(ThreadTeamMode::RUNNING_CLOSED_QUEUE);
        if (errMsg != "") {
            return errMsg;
        }
    }

    return errMsg;
}
}
