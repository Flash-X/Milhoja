#include "ThreadTeamTerminating.h"

/**
 * Instantiate a ThreadTeamTerminating object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamTerminating::ThreadTeamTerminating(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamTerminating::ThreadTeamTerminating] ");
        msg += team_->hdr_;
        msg += "\n\tGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Destroy the concrete state object.
 */
ThreadTeamTerminating::~ThreadTeamTerminating(void) { }

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamTerminating::mode(void) const {
    return ThreadTeam::MODE_TERMINATING;
}

/**
 * 
 */
std::string ThreadTeamTerminating::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    if        (team_->N_wait_ != 0) {
        errMsg = "N_wait not zero";
    } else if (team_->N_comp_ != 0) {
        errMsg = "N_comp not zero";
    } else if (!team_->queue_.empty()) {
        errMsg = "Pending work queue not empty";
    }

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not start a new task if the team is terminating.
 *
 */
void ThreadTeamTerminating::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                      const std::string& teamName, 
                                      const std::string& taskName) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "startTask", 0, "Cannot start a task if team is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not activate threads if the team is terminating.
 */
void ThreadTeamTerminating::increaseThreadCount(const unsigned int nThreads) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "increaseThreadCount", 0,
        "Cannot increase thread count if team is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not allow for work to be added if the team is terminating.
 */
void ThreadTeamTerminating::enqueue(const int work) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "enqueue", 0, "Cannot add more work if team is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * No task can be running if the team is terminating.
 */
void ThreadTeamTerminating::closeTask() {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "closeTask", 0, "Cannot close queue if team is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Don't let a thread wait on this object to finish a task if the team is
 * terminating.
 */
void ThreadTeamTerminating::wait(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "wait", 0, "Cannot wait on team that is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not add a subscriber if the team is terminating.
 */
void ThreadTeamTerminating::attachThreadReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "attachThreadReciever", 0, "Cannot attach to team that is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * A terminating team should not have any subscribers.
 */
void ThreadTeamTerminating::detachThreadReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "detachThreadReciever", 0,
        "No team should be attached to terminating team");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}
    
/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not add a subscriber if the team is terminating.
 */
void ThreadTeamTerminating::attachWorkReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "attachWorkReciever", 0, "Cannot attach to team that is terminating");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * A terminating team should not have any subscribers.
 */
void ThreadTeamTerminating::detachWorkReceiver(void) {
    pthread_mutex_lock(&(team_->teamMutex_));

    std::string  errMsg = team_->printState_NotThreadsafe(
        "detachWorkReciever", 0,
        "No team should be attached to terminating team");

    pthread_mutex_unlock(&(team_->teamMutex_));
    throw std::logic_error(errMsg);
}

