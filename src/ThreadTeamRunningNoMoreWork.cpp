#include "ThreadTeamRunningNoMoreWork.h"

/**
 *
 *
 * \param
 * \return
 */
ThreadTeamRunningNoMoreWork::ThreadTeamRunningNoMoreWork(ThreadTeam* team)
    : ThreadTeamState(),
      team_(team)
{
    if (!team_) {
        std::string  msg("[ThreadTeamRunningNoMoreWork::ThreadTeamRunningNoMoreWork] ");
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
ThreadTeamRunningNoMoreWork::~ThreadTeamRunningNoMoreWork(void) { }

/**
 *
 *
 * \param
 * \return
 */
ThreadTeam::teamMode ThreadTeamRunningNoMoreWork::mode(void) const {
    return ThreadTeam::MODE_RUNNING_NO_MORE_WORK;
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeamRunningNoMoreWork::startTask(TASK_FCN* fcn, const unsigned int nThreads,
                                            const std::string& teamName, 
                                            const std::string& taskName) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::startTask] ");
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
void ThreadTeamRunningNoMoreWork::increaseThreadCount(const unsigned int nThreads) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::increaseThreadCount] ");
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
void ThreadTeamRunningNoMoreWork::enqueue(const int work) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::enqueue] ");
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
void ThreadTeamRunningNoMoreWork::closeTask() {
    std::string  msg("[ThreadTeamRunningNoMoreWork::closeTask] ");
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
void ThreadTeamRunningNoMoreWork::wait(void) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::closeTask] ");
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
void ThreadTeamRunningNoMoreWork::attachThreadReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::attachThreadReceiver] ");
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
void ThreadTeamRunningNoMoreWork::detachThreadReceiver(void) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::detachThreadReceiver] ");
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
void ThreadTeamRunningNoMoreWork::attachWorkReceiver(ThreadTeam* receiver) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::attachWorkReceiver] ");
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
void ThreadTeamRunningNoMoreWork::detachWorkReceiver(void) {
    std::string  msg("[ThreadTeamRunningNoMoreWork::detachWorkReceiver] ");
    msg += team_->hdr_;
    msg += "\nCannot detach from a team that is terminating";
    throw std::logic_error(msg);
}

