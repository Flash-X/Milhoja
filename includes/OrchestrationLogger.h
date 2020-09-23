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

    static Logger& instance(void);
    static void   setLogFilename(const std::string& filename);

    void   log(const std::string& msg) const;

private:
    Logger(void);

    static std::string    logFilename_;

    std::chrono::steady_clock::time_point   startTime_;
};

}

#endif

