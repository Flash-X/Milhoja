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

#include <mpi.h>

namespace milhoja {

class Logger {
public:
    ~Logger(void);

    Logger(Logger&)                  = delete;
    Logger(const Logger&)            = delete;
    Logger(Logger&&)                 = delete;
    Logger& operator=(Logger&)       = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&)      = delete;

    static  void    initialize(const std::string& filename,
                               const MPI_Comm comm,
                               const int logRank);
    static  Logger& instance(void);
    void            finalize(void);

    static void    setLogFilename(const std::string& filename);

    void   log(const std::string& msg) const;

private:
    Logger(void);

    static std::string    logFilename_;
    static MPI_Comm       comm_;
    static int            logRank_;
    static bool           initialized_;
    static bool           finalized_;

    std::chrono::steady_clock::time_point   startTime_;
    int                                     rank_;
};

}

#endif

