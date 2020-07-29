#include "OrchestrationLogger.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace orchestration {

std::string     Logger::logFilename_ = "";

void   Logger::setLogFilename(const std::string& filename) {
    if (filename == "") {
        throw std::logic_error("[Logger::setLogFilename] "
                               "Empty filename given");
    }

    logFilename_ = filename;
}

/**
 * 
 *
 * \return 
 */
Logger& Logger::instance(void) {
    if (logFilename_ == "") {
        std::cerr << "ERROR - set log filename before getting instance\n";
        throw std::logic_error("[Logger::instance] "
                               "Set filename before getting instance");
    }

    static Logger     orSingleton;
    return orSingleton;
}

void   Logger::log(const std::string& msg) const {
    std::ofstream  logFile{};
    logFile.open(logFilename_, std::ios::out | std::ios::app);
    logFile << msg << std::endl;
    logFile.close();
}

}

