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
    return "";
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not start a new task if the team is terminating.
 *
 */
std::string ThreadTeamTerminating::startTask_NotThreadsafe(TASK_FCN* fcn,
                                                           const unsigned int nThreads,
                                                           const std::string& teamName, 
                                                           const std::string& taskName) {
    return team_->printState_NotThreadsafe("startTask", 0,
                  "Cannot start a task if team is terminating");
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
std::string ThreadTeamTerminating::enqueue_NotThreadsafe(const int work) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot add more work if team is terminating");
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
std::string ThreadTeamTerminating::wait_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("wait", 0,
                  "Cannot wait on team that is terminating");
}

