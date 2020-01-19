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
        msg += "\nGiven thread team in NULL";
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
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not start a new task if the team is terminating.
 *
 */
void ThreadTeamTerminating::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                      const std::string& teamName, 
                                      const std::string& taskName) {
    std::string  msg("[ThreadTeamTerminating::startTask] ");
    msg += team_->hdr_;
    msg += "\nCannot start a task with a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not activate threads if the team is terminating.
 */
void ThreadTeamTerminating::increaseThreadCount(const unsigned int nThreads) {
    std::string  msg("[ThreadTeamTerminating::increaseThreadCount] ");
    msg += team_->hdr_;
    msg += "\nCannot increase thread count in a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not allow for work to be added if the team is terminating.
 */
void ThreadTeamTerminating::enqueue(const int work) {
    std::string  msg("[ThreadTeamTerminating::enqueue] ");
    msg += team_->hdr_;
    msg += "\nCannot add give work to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * No task can be running if the team is terminating.
 */
void ThreadTeamTerminating::closeTask() {
    std::string  msg("[ThreadTeamTerminating::closeTask] ");
    msg += team_->hdr_;
    msg += "\nCannot close the queue of a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Don't let a thread wait on this object to finish a task if the team is
 * terminating.
 */
void ThreadTeamTerminating::wait(void) {
    std::string  msg("[ThreadTeamTerminating::wait] ");
    msg += team_->hdr_;
    msg += "\nCannot wait on a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not add a subscriber if the team is terminating.
 */
void ThreadTeamTerminating::attachThreadReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamTerminating::attachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * A terminating team should not have any subscribers.
 */
void ThreadTeamTerminating::detachThreadReceiver(void) {
    std::string  msg("[ThreadTeamTerminating::detachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nNo team should be attached as a subscriber to a terminating team";
    throw std::logic_error(msg);
}
    
/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not add a subscriber if the team is terminating.
 */
void ThreadTeamTerminating::attachWorkReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamTerminating::attachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * A terminating team should not have any subscribers.
 */
void ThreadTeamTerminating::detachWorkReceiver(void) {
    std::string  msg("[ThreadTeamTerminating::detachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nNo team should be attached as a subscriber to a terminating team";
    throw std::logic_error(msg);
}

