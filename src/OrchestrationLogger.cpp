#include "OrchestrationLogger.h"

#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace orchestration {

std::string     Logger::logFilename_ = "";
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
void   Logger::instantiate(const std::string& filename) {
    if (instantiated_) {
        throw std::logic_error("[Logger::instantiate] Already instantiated");
    }

    setLogFilename(filename);
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
    : startTime_{std::chrono::steady_clock::now()}
{
    using namespace std::chrono;

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
}

/**
 * 
 *
 */
void   Logger::log(const std::string& msg) const {
    using seconds = std::chrono::duration<double>;

    auto endTime = std::chrono::steady_clock::now();
    std::string   elapsedTime = std::to_string(seconds(endTime - startTime_).count());
    elapsedTime += " s - ";

    std::ofstream  logFile{};
    logFile.open(logFilename_, std::ios::out | std::ios::app);
    logFile << elapsedTime << msg << std::endl;
    logFile.close();
}

}

