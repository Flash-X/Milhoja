#include "ThreadTeamRunningClosed.h"

/**
 *
 *
 * \param
 * \return
 */
ThreadTeamRunningClosed::ThreadTeamRunningClosed(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningClosed::ThreadTeamRunningClosed] ");
        msg += team_->hdr_;
        msg += "\nGiven thread team in NULL";
        throw std::logic_error(msg);
    }
}

/**
 *
 *
 * \param
 * \return
 */
ThreadTeamRunningClosed::~ThreadTeamRunningClosed(void) { }

/**
 *
 *
 * \param
 * \return
 */
ThreadTeam::teamMode ThreadTeamRunningClosed::mode(void) const {
    return ThreadTeam::MODE_RUNNING_CLOSED_QUEUE;
}

/**
 * 
 */
std::string ThreadTeamRunningClosed::isStateValid_NotThreadSafe(void) const {
    std::string errMsg("");

    return errMsg;
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                        const std::string& teamName, 
                                        const std::string& taskName) {
    std::string  msg("[ThreadTeamRunningClosed::startTask] ");
    msg += team_->hdr_;
    msg += "\nCannot start a task with a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::increaseThreadCount(const unsigned int nThreads) {
    std::string  msg("[ThreadTeamRunningClosed::increaseThreadCount] ");
    msg += team_->hdr_;
    msg += "\nCannot increase thread count in a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::enqueue(const int work) {
    std::string  msg("[ThreadTeamRunningClosed::enqueue] ");
    msg += team_->hdr_;
    msg += "\nCannot add give work to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::closeTask() {
    std::string  msg("[ThreadTeamRunningClosed::closeTask] ");
    msg += team_->hdr_;
    msg += "\nCannot close the queue of a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::wait(void) {
    std::string  msg("[ThreadTeamRunningClosed::closeTask] ");
    msg += team_->hdr_;
    msg += "\nCannot wait on a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::attachThreadReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningClosed::attachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::detachThreadReceiver(void) {
    std::string  msg("[ThreadTeamRunningClosed::detachThreadReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot detach from a team that is terminating";
    throw std::logic_error(msg);
}
    
/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::attachWorkReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningClosed::attachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot attach to a team that is terminating";
    throw std::logic_error(msg);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningClosed::detachWorkReceiver(void) {
    std::string  msg("[ThreadTeamRunningClosed::detachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot detach from a team that is terminating";
    throw std::logic_error(msg);
}

