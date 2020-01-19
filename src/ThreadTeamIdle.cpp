#include "ThreadTeamIdle.h"

#include <iostream>

/**
 * Instantiate a ThreadTeamIdle object for internal use by a ThreadTeam object
 * as part of the State design pattern.  This gives the concrete state object a
 * pointer to the ThreadTeam object whose private data members it will directly
 * adjust under the hood.
 *
 * \todo - Finish implementing all methods
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
        msg += "\nGiven thread team in NULL";
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
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamIdle::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                               const std::string& teamName, 
                               const std::string& taskName) {
    std::string  msg("[ThreadTeamIdle::startTask] ");
    msg += team_->hdr_;
    msg += "\nstartTask() not implemented yet for mode Idle";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Pass the threads along to a thread subscriber if the object has one.
 * Otherwise, it does nothing.  This behavior is motivated by the case that a
 * Thread Subscriber has gone Idle before its Thread Publisher has finished its
 * task.  Note that all subsequent teams that are Idle and that receive this
 * signal will ignore it as well as adding threads to a team with no task is
 * non-sensical.
 */
void ThreadTeamIdle::increaseThreadCount(const unsigned int nThreads) {
    pthread_mutex_lock(&(team_->teamMutex_));

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
    std::string  msg("[ThreadTeamIdle::enqueue] ");
    msg += team_->hdr_;
    msg += "\nAdding work in Idle state not allowed";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Calling this is a logic error as we cannot close the queue if no task is
 * running.
 */
void ThreadTeamIdle::closeTask() {
    std::string  msg("[ThreadTeamIdle::closeTask] ");
    msg += team_->hdr_;
    msg += "\nCannot close the queue when no task is being executed";
    throw std::logic_error(msg);
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
#ifdef VERBOSE
    // DEBUG - Does this ever happen?
    pthread_mutex_lock(&(team_->teamMutex_));

    if (team_->isWaitBlocking_) {
        std::string  errMsg("ThreadTeamIdle::wait] ");
        errMsg += team_->hdr_;
        errMsg += "\nTeam incorrectly believes that a thread already called wait";
        throw std::runtime_error(errMsg);
    }

    std::cout << "[" << team_->hdr_ << "] wait() called with team in Idle\n";
    pthread_mutex_unlock(&(team_->teamMutex_));
#endif
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

    if (!receiver) {
        std::string  msg("[ThreadTeamIdle::attachThreadReceiver] ");
        msg += team_->hdr_;
        msg += "\nNull Thread Subscriber team given";
        throw std::logic_error(msg);
    } else if (receiver == team_) {
        std::string  msg("[ThreadTeamIdle::attachThreadReceiver] ");
        msg += team_->hdr_;
        msg += "\nCannot attach the team to itself";
        throw std::logic_error(msg);
    } else if (team_->threadReceiver_) {
        std::string  msg("[ThreadTeamIdle::attachThreadReceiver] ");
        msg += team_->hdr_;
        msg += "\nA Thread Subscriber is already attached";
        throw std::logic_error(msg);
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

    if (!(team_->threadReceiver_)) {
        std::string  msg("[ThreadTeamIdle::detachThreadReceiver] ");
        msg += team_->hdr_;
        msg += "\nNo Thread Subscriber attached";
        throw std::logic_error(msg);
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

    if (!receiver) {
        std::string  msg("[ThreadTeamIdle::attachWorkReceiver] ");
        msg += team_->hdr_;
        msg += "\nNull Work Subscriber team given";
        throw std::logic_error(msg);
    } else if (receiver == team_) {
        std::string  msg("[ThreadTeamIdle::attachWorkReceiver] ");
        msg += team_->hdr_;
        msg += "\nCannot attach the team to itself";
        throw std::logic_error(msg);
    } else if (team_->workReceiver_) {
        std::string  msg("[ThreadTeamIdle::attachWorkReceiver] ");
        msg += team_->hdr_;
        msg += "\nA Work Subscriber is already attached";
        throw std::logic_error(msg);
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

    if (!(team_->workReceiver_)) {
        std::string  msg("[ThreadTeamIdle::detachWorkReceiver] ");
        msg += team_->hdr_;
        msg += "\nNo Work Subscriber attached";
        throw std::logic_error(msg);
    }
    team_->workReceiver_ = nullptr;

    pthread_mutex_unlock(&(team_->teamMutex_));
}

