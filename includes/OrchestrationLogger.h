/**
 * \file    OrchestrationLogger.h
 *
 * \brief 
 *
 * MPI should be initialized before instantiating the logger.
 *
 */

#ifndef ORCHESTRATION_LOGGER_H__
#define ORCHESTRATION_LOGGER_H__

#include <string>
#include <chrono>

#ifndef LOGGER_NO_MPI
#include <mpi.h>
#endif

namespace orchestration {

class Logger {
public:
    ~Logger(void);

    Logger(Logger&)                  = delete;
    Logger(const Logger&)            = delete;
    Logger(Logger&&)                 = delete;
    Logger& operator=(Logger&)       = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&)      = delete;

#ifdef LOGGER_NO_MPI
    static void    instantiate(const std::string& filename);
#else
    static void    instantiate(const MPI_Comm comm, const std::string& filename);
#endif
    static Logger& instance(void);
    static void    setLogFilename(const std::string& filename);

    void   log(const std::string& msg) const;

private:
    Logger(void);

    static std::string    logFilename_;
#ifndef LOGGER_NO_MPI
    static MPI_Comm       globalComm_;
#endif
    static bool           instantiated_;

    std::chrono::steady_clock::time_point   startTime_;
    int                                     rank_;
};

}

#endif

