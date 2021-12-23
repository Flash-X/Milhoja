/**
 * \file Timer.h
 *
 * \brief A collection of functions for timing long-duration events in a
 * low-precision way.
 *
 * The tools in this collection are intended for coarse, aggregated timing of
 * major program effort in a low-precision way.  By aggregate, we mean that the
 * data is used to determine the total walltime for executing a certain type
 * of computation (e.g. the total walltime across all GC fills).
 *
 * The start and stop functions have the same single MPI process write to the
 * program's log file a line indicating that the timer was started or stopped.
 * As such, the times at which start and stop are called are written to file.
 * Each code block to be timed should be given a unique name, which is passed to
 * these functions.  Therefore, for each start call, there should be a single
 * subsequent call to stop using the same name.  While possible, it is a logical
 * error to call start and stop out of order with the same name --- they should
 * be called as a pair.
 *
 * Note that nested use of the timing facility is possible.  Also, the same code
 * block might be executed multiple times, in which case multiple start/stop
 * timestamp pairs will appear in the log file.  These pairs can then be
 * extracted in post-processing to determine how long the code section took to
 * run with each execution.  In addition, the length of all such related
 * intervals can be summed.  Note that the logging of the start and stop times
 * is done in the precision of the Logger class, which sets the lower bound on
 * the timer resolution.
 *
 * Note that before logging, the start or stop functions call an MPI barrier.
 * Therefore, all MPI processes in the global communicator must call the
 * functions.  Also, this implies that the time intervals effectively measured
 * by the single MPI process are the maximum walltime across all MPI processes.
 */

#ifndef TIMER_H__
#define TIMER_H__

#include <string>

#include <mpi.h>

#include <Milhoja_Logger.h>

#include "Sedov.h"

namespace Timer {
    inline void   start(const std::string& msg) {
        MPI_Barrier(GLOBAL_COMM);
        milhoja::Logger::instance().log("[Timer] Start timing " + msg);
    }

    inline void   stop(const std::string& msg) {
        MPI_Barrier(GLOBAL_COMM);
        milhoja::Logger::instance().log("[Timer] Stop timing " + msg);
    }
}

#endif

