#include "Milhoja_Logger.h"
#include "Milhoja.h"

#include <ctime>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace milhoja {

std::string     Logger::logFilename_ = "";
MPI_Comm        Logger::comm_        = MPI_COMM_NULL;
int             Logger::logRank_     = -1;
bool            Logger::initialized_ = false;
bool            Logger::finalized_   = false;

/**
 * In most cases, the log will be written to a single log file throughout the entire
 * application execution.  Therefore, setting the filename via initialize
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
        throw std::invalid_argument("[Logger::setLogFilename] Empty filename given");
    }

    logFilename_ = filename;
}

/**
 * 
 *
 * \return 
 */
Logger& Logger::instance(void) {
    if        (!initialized_) {
#ifdef RUNTIME_MUST_USE_TILEITER
        throw std::logic_error("[Logger::instance] Singleton not initialized");
#else
        initialize("MILHOJA.log", MPI_COMM_WORLD, 0);//lazy init, only with defaults
#endif
    }
    if (finalized_) {
        throw std::logic_error("[Logger::instance] No access after finalization");
    }

    static Logger     singleton;
    return singleton;
}

/**
 * Instantiate the logging singleton, which persists until program termination.
 * This should only be called once and must be called before any calling code
 * attempts to access the singleton via the instance method.
 *
 * \param filename - the name of the file to which log output should be written
 * \param comm - the MPI communicator to be used to identify the MPI logging
 *               process
 * \param logRank - the rank of the single MPI process that should perform logging
 */
void   Logger::initialize(const std::string& filename,
                          const MPI_Comm comm,
                          const int logRank) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[Logger::initialize] Already initialized");
    }
    setLogFilename(filename);
    
    int     wasInitialized = false;
    MPI_Initialized(&wasInitialized);

    if (!wasInitialized) {
        throw std::logic_error("[Logger::initialize] Please initialize MPI first");
    } else if (logRank < 0) {
        throw std::invalid_argument("[Logger::initialize] Negative rank");
    } else if (comm == MPI_COMM_NULL) {
        throw std::invalid_argument("[Logger::initialize] Null MPI communicator");
    }

    comm_ = comm;
    logRank_ = logRank;

    initialized_ = true;
    instance();
}

/**
 *
 */
void    Logger::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[Logger::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[Logger::finalize] Already finalized");
    }

    auto         now   = std::chrono::system_clock::now();
    std::time_t  now_t = std::chrono::system_clock::to_time_t(now);
    char  timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%FT%T", std::gmtime(&now_t));
    log("[Logger] Terminated at " + std::string(timestamp) + " UTC");

    logFilename_ = "";
    comm_        = MPI_COMM_NULL;
    logRank_     = -1;
    rank_        = -1;

    finalized_ = true;
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

    MPI_Comm_rank(comm_, &rank_);

    if (rank_ == logRank_) {
        // Get time to be used for elapsed time
        auto         now   = system_clock::now();
        std::time_t  now_t = system_clock::to_time_t(now);
        char  timestamp[100];
        std::strftime(timestamp, sizeof(timestamp), "%FT%T", std::gmtime(&now_t));
        std::string   msg{};
        log("[Logger] Started at " + std::string(timestamp) + " UTC");
    }
}

/**
 *
 */
Logger::~Logger(void) {
    if (initialized_ && !finalized_) {
#ifdef RUNTIME_MUST_USE_TILEITER
        // The Milhoja grid should have closed the logger on finalization,
        // complain a bit if not.
        std::cerr << "[Logger::~Logger] ERROR - Not finalized" << std::endl;
#else
        log("[Logger] ~Logger called, high time to terminate:");
        finalize();
#endif
    }
}

/**
 * 
 *
 */
void   Logger::log(const std::string& msg) const {
    using seconds = std::chrono::duration<double>;

    // We don't need to check if initialized, because this cannot be called
    // without first calling instance.
    if (finalized_) {
        throw std::logic_error("[Logger::log] Cannot use after finalization");
    }

    if (rank_ == logRank_) {
        auto endTime = std::chrono::steady_clock::now();
        std::string   elapsedTime = std::to_string(seconds(endTime - startTime_).count());
        elapsedTime += " s - ";

        std::ofstream  logFile{};
        logFile.open(logFilename_, std::ios::out | std::ios::app);
        logFile << elapsedTime << msg << std::endl;
        logFile.close();
    }
}

}

