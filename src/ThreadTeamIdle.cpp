#include "ThreadTeamIdle.h"
namespace orchestration {

/**
 * Instantiate a ThreadTeamIdle object for internal use by a ThreadTeam object
 * as part of the State design pattern.  This gives the concrete state object a
 * pointer to the ThreadTeam object whose private data members it will directly
 * adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
template<typename W, class T>
ThreadTeamIdle<W,T>::ThreadTeamIdle(T* team)
    : ThreadTeamState<W,T>(),
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
 * Confirm that the state of the EFSM is valid for the Idle mode.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamIdle<W,T>::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if (team_->N_idle_ != team_->nMaxThreads_) {
        errMsg  = "N_idle != N threads in team";
    } else if (team_->N_wait_ != 0) {
        errMsg = "N_wait not zero";
    } else if (team_->N_comp_ != 0) {
        errMsg = "N_comp not zero";
    } else if (team_->N_terminate_ != 0) {
        errMsg = "N_terminate not zero";
    } else if (!(team_->queue_.empty())) {
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
template<typename W, class T>
std::string ThreadTeamIdle<W,T>::startTask_NotThreadsafe(const RuntimeAction& action,
                                                         const std::string& teamName) {
    std::string   errMsg("");

#ifdef DEBUG_RUNTIME
    team_->logFile_.open(team_->logFilename_, std::ios::out | std::ios::app);
    team_->logFile_ << "[" << team_->hdr_ << "] Assigned team name "
                    << teamName << std::endl;
    team_->logFile_ << "[" << teamName << "] Starting task "
                    << action.name
                    << " with "
                    << std::to_string(action.nInitialThreads)
                    << " initial threads\n";
    team_->logFile_.close();
#endif

    team_->hdr_ = teamName;
    team_->taskFcn_ = action.routine;

    // Timing tests originally failed on occasion when a single thread no-op
    // cycle was run after many 10 thread no-op cycles.  The execution cycles
    // were so fast that they were finishing before many threads had the chance
    // to be activated.  Eventually, there were more pending threads than
    // requested threads when startTask was called on the single thread cycle.
    while (team_->N_to_activate_ > action.nInitialThreads) {
        pthread_cond_wait(&(team_->allActivated_), &(team_->teamMutex_));
    }
    // This cannot rollover
    unsigned int nEventsToEmit = action.nInitialThreads - team_->N_to_activate_;

    team_->N_to_activate_ = action.nInitialThreads;
    errMsg = team_->setMode_NotThreadsafe(ThreadTeamMode::RUNNING_OPEN_QUEUE);
    if (errMsg != "") {
        return errMsg;
    }

    for (unsigned int i=0; i<nEventsToEmit; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    return errMsg;
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
template<typename W, class T>
std::string ThreadTeamIdle<W,T>::increaseThreadCount_NotThreadsafe(
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
template<typename W, class T>
std::string  ThreadTeamIdle<W,T>::enqueue_NotThreadsafe(W& work, const bool move) {
    // TODO: Consider (carefully!) allowing for enqueueing of data items when
    // when the team is Idle and with a routine that will only acquire the mutex
    // once.  This could allow for decreasing mutex contention when threads
    // become active.  We want them to find work immediately and not box out the
    // thread trying to give them work.
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
template<typename W, class T>
std::string ThreadTeamIdle<W,T>::closeTask_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeTask", 0,
                  "Cannot close the queue when no task is being executed");
}
}
