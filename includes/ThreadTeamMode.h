/**
 * \file    ThreadTeamMode.h
 *
 * The qualitative modes that are used as part of the full state definition
 * of the ThreadTeam extended finite state machine.  Note that Terminating is
 * not a mode in the EFSM design.  However, we implement it as if it were to
 * handle the termination of the EFSM as part of the State design pattern.
 *
 * This is defined separately from the ThreadTeam class definition as the modes
 * are not dependent on the template parameters of that class.
 */

#ifndef THREAD_TEAM_MODES_H__
#define THREAD_TEAM_MODES_H__

enum class ThreadTeamMode {IDLE,
                           TERMINATING,
                           RUNNING_OPEN_QUEUE,
                           RUNNING_CLOSED_QUEUE,
                           RUNNING_NO_MORE_WORK};

#endif

