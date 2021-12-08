/**
 * \file    OrchestrationLogger.h
 *
 * \brief 
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
    static void    instantiate(const std::string& filename,
                               const MPI_Comm comm, const int logRank);
#endif
    static Logger& instance(void);
    static void    setLogFilename(const std::string& filename);

#ifndef LOGGER_NO_MPI
    void   acquireRank(void);
#endif

    void   log(const std::string& msg) const;

private:
    Logger(void);

    static std::string    logFilename_;
#ifndef LOGGER_NO_MPI
    static MPI_Comm       comm_;
    static int            logRank_;
#endif
    static bool           instantiated_;

    std::chrono::steady_clock::time_point   startTime_;
    int                                     rank_;
};

}

#endif

