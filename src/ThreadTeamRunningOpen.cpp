#include "ThreadTeamRunningOpen.h"

/**
 * Instantiate a ThreadTeamRunningOpen object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
ThreadTeamRunningOpen::ThreadTeamRunningOpen(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningOpen::ThreadTeamRunningOpen] ");
        msg += team_->hdr_;
        msg += "\nGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 * Destroy the concrete state object.
 */
ThreadTeamRunningOpen::~ThreadTeamRunningOpen(void) { }

/**
 * Obtain the mode that this class is associated with.
 *
 * \return The mode as a value in the teamMode enum.
 */
ThreadTeam::teamMode ThreadTeamRunningOpen::mode(void) const {
    return ThreadTeam::MODE_RUNNING_OPEN_QUEUE;
}

/**
 * 
 */
std::string ThreadTeamRunningOpen::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    return errMsg;
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                      const std::string& teamName, 
                                      const std::string& taskName) {
    std::string  msg("[ThreadTeamRunningOpen::startTask] ");
    msg += team_->hdr_;
    msg += "\nCannot start a task with a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::increaseThreadCount(const unsigned int nThreads) {
    std::string  msg("[ThreadTeamRunningOpen::increaseThreadCount] ");
    msg += team_->hdr_;
    msg += "\nCannot increase thread count in a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::enqueue(const int work) {
    std::string  msg("[ThreadTeamRunningOpen::enqueue] ");
    msg += team_->hdr_;
    msg += "\nCannot add give work to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::closeTask() {
    std::string  msg("[ThreadTeamRunningOpen::closeTask] ");
    msg += team_->hdr_;
    msg += "\nCannot close the queue of a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::wait(void) {
    std::string  msg("[ThreadTeamRunningOpen::wait] ");
    msg += team_->hdr_;
    msg += "\nCannot wait on a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::attachThreadReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningOpen::attachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::detachThreadReceiver(void) {
    std::string  msg("[ThreadTeamRunningOpen::detachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot detach from a team that is terminating";
    throw std::logic_error(msg);
}
    
/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::attachWorkReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningOpen::attachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 */
void ThreadTeamRunningOpen::detachWorkReceiver(void) {
    std::string  msg("[ThreadTeamRunningOpen::detachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot detach from a team that is terminating";
    throw std::logic_error(msg);
}

