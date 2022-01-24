/**
 * \file    Milhoja_Logger.h
 *
 * \brief 
 *
 */

#ifndef MILHOJA_LOGGER_H__
#define MILHOJA_LOGGER_H__

#include <string>
#include <chrono>

#ifndef LOGGER_NO_MPI
#include <mpi.h>
#endif

namespace milhoja {

class Logger {
public:
    ~Logger(void)   {}

    Logger(Logger&)                  = delete;
    Logger(const Logger&)            = delete;
    Logger(Logger&&)                 = delete;
    Logger& operator=(Logger&)       = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&)      = delete;

#ifdef LOGGER_NO_MPI
    static  void    initialize(const std::string& filename);
#else
    static  void    initialize(const std::string& filename,
                               const MPI_Comm comm,
                               const int logRank);
#endif
    static  Logger& instance(void);
    void            finalize(void);

    static void    setLogFilename(const std::string& filename);

    void   log(const std::string& msg) const;

private:
    Logger(void);

    static std::string    logFilename_;
#ifndef LOGGER_NO_MPI
    static MPI_Comm       comm_;
    static int            logRank_;
#endif
    static bool           initialized_;
    static bool           finalized_;

    std::chrono::steady_clock::time_point   startTime_;
    int                                     rank_;
};

}

#endif

