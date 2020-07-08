#include "ThreadTeamTerminating.h"
namespace orchestration {

/**
 * Instantiate a ThreadTeamTerminating object for internal use by a ThreadTeam
 * object as part of the State design pattern.  This gives the concrete state
 * object a pointer to the ThreadTeam object whose private data members it will
 * directly adjust under the hood.
 *
 * \param team - The ThreadTeam object that is instantiating this object
 */
template<typename W, class T>
ThreadTeamTerminating<W,T>::ThreadTeamTerminating(T* team)
    : ThreadTeamState<W,T>(),
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
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not start a new task if the team is terminating.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamTerminating<W,T>::startTask_NotThreadsafe(const RuntimeAction& action,
                                                                const std::string& teamName) {
    return team_->printState_NotThreadsafe("startTask", 0,
                  "Cannot start a task if team is terminating");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not activate threads if the team is terminating.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamTerminating<W,T>::increaseThreadCount_NotThreadsafe(
                                            const unsigned int nThreads) {
    return team_->printState_NotThreadsafe("increaseThreadCount", 0,
        "Cannot increase thread count if team is terminating");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * Do not allow for work to be added if the team is terminating.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamTerminating<W,T>::enqueue_NotThreadsafe(W& work, const bool move) {
    return team_->printState_NotThreadsafe("enqueue", 0,
                  "Cannot add more work if team is terminating");
}

/**
 * See ThreadTeam.cpp documentation for same method for basic information.
 *
 * No task can be running if the team is terminating.
 *
 * \warning This method is *not* thread safe and therefore should only be called
 *          when the calling code has already acquired teamMutex_.
 *
 * \return an empty string if the state is valid.  Otherwise, an error message
 */
template<typename W, class T>
std::string ThreadTeamTerminating<W,T>::closeTask_NotThreadsafe(void) {
    return team_->printState_NotThreadsafe("closeTask", 0,
                  "Cannot close queue if team is terminating");
}
}
