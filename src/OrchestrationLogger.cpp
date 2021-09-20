#include "OrchestrationLogger.h"

#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "constants.h"

namespace orchestration {

std::string     Logger::logFilename_ = "";
#ifndef LOGGER_NO_MPI
MPI_Comm        Logger::globalComm_ = MPI_COMM_NULL;
#endif
bool            Logger::instantiated_ = false;

/**
 * In most cases, the log will be written to a single log file throughout the entire
 * application execution.  Therefore, setting the filename via instantiate
 * should be sufficient.
 *
 * This method is provided so that each test in a test suite can write their
 * results to a dedicated log file.
 *
 * \param filename - the name of the file to which logging should be written.
 *                   An empty value is not acceptable.
 */
void   Logger::setLogFilename(const std::string& filename) {
    if (filename == "") {
        throw std::logic_error("[Logger::setLogFilename] Empty filename given");
    }

    logFilename_ = filename;
}

/**
 * 
 *
 */
#ifdef LOGGER_NO_MPI
void   Logger::instantiate(const std::string& filename) {
#else
void   Logger::instantiate(const MPI_Comm comm, const std::string& filename) {
#endif
    if (instantiated_) {
        throw std::logic_error("[Logger::instantiate] Already instantiated");
    }

    setLogFilename(filename);
#ifndef LOGGER_NO_MPI
    globalComm_ = comm;
#endif
    instantiated_ = true;

    instance();
}

/**
 * 
 *
 * \return 
 */
Logger& Logger::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[Logger::instance] "
                               "Logger must be instantiated first");
    }

    static Logger     singleton;
    return singleton;
}

/**
 * 
 *
 */
Logger::Logger(void)
    : startTime_{std::chrono::steady_clock::now()},
      rank_{-1}
{
    using namespace std::chrono;

#ifndef LOGGER_NO_MPI
    MPI_Comm_rank(globalComm_, &rank_);
#endif

    // Get time to be used for elapsed time
    auto         now   = system_clock::now();
    std::time_t  now_t = system_clock::to_time_t(now);
    char  timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%FT%T", std::gmtime(&now_t));
    std::string   msg{};
    log("[Logger] Started at " + std::string(timestamp) + " UTC");
}

/**
 * 
 *
 */
Logger::~Logger(void) {
    auto         now   = std::chrono::system_clock::now();
    std::time_t  now_t = std::chrono::system_clock::to_time_t(now);
    char  timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%FT%T", std::gmtime(&now_t));
    log("[Logger] Terminated at " + std::string(timestamp) + " UTC");

    instantiated_ = false;
    rank_ = -1;
}

/**
 * 
 *
 */
void   Logger::log(const std::string& msg) const {
    using seconds = std::chrono::duration<double>;

#ifndef LOGGER_NO_MPI
    if (rank_ == MASTER_PE) {
#endif
        auto endTime = std::chrono::steady_clock::now();
        std::string   elapsedTime = std::to_string(seconds(endTime - startTime_).count());
        elapsedTime += " s - ";

        std::ofstream  logFile{};
        logFile.open(logFilename_, std::ios::out | std::ios::app);
        logFile << elapsedTime << msg << std::endl;
        logFile.close();
#ifndef LOGGER_NO_MPI
    }
#endif
}

}

