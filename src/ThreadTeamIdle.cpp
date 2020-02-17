#include "ThreadTeamIdle.h"

#include <iostream>

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
 * Destroy the Idle concrete state object.
 */
ThreadTeamIdle::~ThreadTeamIdle(void) { }

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
 */
void ThreadTeamIdle::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                               const std::string& teamName, 
                               const std::string& taskName) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "startTask", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (nThreads > (team_->N_idle_ - team_->N_to_activate_)) {
        msg  = "nThreads (";
        msg += std::to_string(nThreads);
        msg += ") exceeds the number of threads available for activation";
        std::string  errMsg = team_->printState_NotThreadsafe(
            "startTask", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (team_->N_to_activate_ != 0) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "startTask", 0, "Number of threads pending activation not zero");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    team_->N_to_activate_ = nThreads;
    team_->setMode_NotThreadsafe(ThreadTeam::MODE_RUNNING_OPEN_QUEUE);
    for (unsigned int i=0; i<nThreads; ++i) {
        pthread_cond_signal(&(team_->activateThread_));
    }

    pthread_mutex_unlock(&(team_->teamMutex_));
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
 */
void ThreadTeamIdle::increaseThreadCount(const unsigned int nThreads) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "startTask", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (nThreads > (team_->N_idle_ - team_->N_to_activate_)) {
        // Even though we aren't activating threads in the team, this still
        // represents a logical error in the program.
        msg  = "nThreads (";
        msg += std::to_string(nThreads);
        msg += ") exceeds the number of threads available for activation";
        std::string  errMsg = team_->printState_NotThreadsafe(
            "startTask", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    // No need to alter N_to_activate_ as we forward the thread activation on
    if (team_->threadReceiver_) {
        team_->threadReceiver_->increaseThreadCount(nThreads);
    }

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * To simplify the design, we do not allow for adding work in Idle mode.  Client
 * code should therefore call startTask() on all ThreadTeams before enqueueing
 * work in any one of them.  In this way, we don't have to worry about an active
 * Work Publisher enqueueing work on a Work Subscriber that is still Idle.
 */
void ThreadTeamIdle::enqueue(const int work) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "enqueue", 0, "Adding work in Idle state not allowed");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Calling this is a logic error as we cannot close the queue if no task is
 * running.
 */
void ThreadTeamIdle::closeTask() {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "enqueue", 0, "Cannot close the queue when no task is being executed");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Calling wait on a ThreadTeam that is Idle seems like a logic error.  However,
 * it could be that a team finishes its task and transition to Idle before a
 * calling thread got a chance to call wait().  Therefore, this method is a
 * no-op so that it won't block.
 */
void ThreadTeamIdle::wait(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "wait", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (team_->isWaitBlocking_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "wait", 0, "Team incorrectly believes that a thread already called wait");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    }

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * It is a logic error
 *   - to call this with a NULL team
 *   - to try to attach a team to itself
 *   - to try to attach when a team is already attached.
 *
 * If no logic error occurs, then the team can be attached in Idle.
 */
void ThreadTeamIdle::attachThreadReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachThreadReceiver", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (!receiver) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachThreadReceiver", 0, "Null thread subscriber team given");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (receiver == team_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachThreadReceiver", 0, "Cannot attach team to itself");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (team_->threadReceiver_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachThreadReceiver", 0, "A thread subscriber is already attached");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    team_->threadReceiver_ = receiver;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * It is a logic error to call this if no team is attached.
 */
void ThreadTeamIdle::detachThreadReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "detachThreadReceiver", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (!(team_->threadReceiver_)) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "detachThreadReceiver", 0, "No thread subscriber attached");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    team_->threadReceiver_ = nullptr;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * It is a logic error
 *   - to call this with a NULL team
 *   - to try to attach a team to itself
 *   - to try to attach when a team is already attached.
 *
 * If no logic error occurs, then the team can be attached in Idle.
 */
void ThreadTeamIdle::attachWorkReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachWorkReceiver", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (!receiver) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachWorkReceiver", 0, "Null work subscriber team given");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (receiver == team_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachWorkReceiver", 0, "Cannot attach team to itself");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    } else if (team_->workReceiver_) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "attachWorkReceiver", 0, "A work subscriber is already attached");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    team_->workReceiver_ = receiver;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * It is a logic error to call this if no team is attached.
 */
void ThreadTeamIdle::detachWorkReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string msg = isStateValid_NotThreadSafe();
    if (msg != "") {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "detachWorkReceiver", 0, msg);
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::runtime_error(errMsg);
    } else if (!(team_->workReceiver_)) {
        std::string  errMsg = team_->printState_NotThreadsafe(
            "detachWorkReceiver", 0, "No work subscriber attached");
        pthread_mutex_unlock(&(team_->teamMutex_));
        throw std::logic_error(errMsg);
    }

    team_->workReceiver_ = nullptr;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

