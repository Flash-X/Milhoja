#include "Milhoja_RuntimeBackend.h"

#include <stdexcept>

#include "Milhoja_Logger.h"
#ifdef MILHOJA_USE_CUDA_BACKEND
#include "Milhoja_CudaBackend.h"
#else
#include "Milhoja_NullBackend.h"
#endif

namespace milhoja {

bool           RuntimeBackend::initialized_ = false;
bool           RuntimeBackend::finalized_   = false;
unsigned int   RuntimeBackend::nStreams_ = 0;
std::size_t    RuntimeBackend::nBytesInMemoryPools_ = 0;

/**
 * Provide the Singleton design pattern with the runtime parameters needed to
 * appropriately instantiate and initialize the desired concrete RuntimeBackend
 * class.  This function must be called before using instance() to gain access
 * to the Singleton object.
 *
 * @param nStreams - the maximum number of streams that the runtime is allowed
 * to use at any point in time.
 * @param nBytesInMemoryPools - the amount of memory to allocate in memory
 * pools.  Refer to the documentation for each backend to determine what memory
 * pools will be allocated.
 */
void   RuntimeBackend::initialize(const unsigned int nStreams,
                            const std::size_t  nBytesInMemoryPools) {
    Logger::instance().log("[RuntimeBackend] Initializing...");

    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[RuntimeBackend::initialize] Already initialized");
    }

    // The responsibility of error checking these falls on the code that
    // consumes them later.
    nStreams_ = nStreams;
    nBytesInMemoryPools_ = nBytesInMemoryPools;

    initialized_ = true;

    // Create/initialize backend
    instance();

    Logger::instance().log("[RuntimeBackend] Created and ready for use");
}

/**
 *
 * Derived classes that overload this implementation should call this member
 * function *after* performing its own clean-up.
 */
void   RuntimeBackend::finalize(void) {
    Logger::instance().log("[RuntimeBackend] Finalizing ...");

    if        (!initialized_) {
        throw std::logic_error("[RuntimeBackend::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[RuntimeBackend::finalize] Already finalized");
    }

    nStreams_ = 0;
    nBytesInMemoryPools_ = 0;

    finalized_ = true;

    Logger::instance().log("[RuntimeBackend] Finalized");
}

/**
 * Obtain access to the RuntimeBackend Singleton object.
 */
RuntimeBackend&   RuntimeBackend::instance(void) {
    if(!initialized_) {
        throw std::logic_error("[RuntimeBackend::instance] initialize first");
    } else if (finalized_) {
        throw std::logic_error("[RuntimeBackend::instance] No access after finalization");
    }

#ifdef MILHOJA_USE_CUDA_BACKEND
    static CudaBackend singleton{nStreams_, nBytesInMemoryPools_};
#else
    static NullBackend singleton{};
    Logger::instance().log("[RuntimeBackend] No runtime backend needed");
#endif

    return singleton;
}

RuntimeBackend::~RuntimeBackend(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[RuntimeBackend::~RuntimeBackend] ERROR - Not finalized" << std::endl;
    }
}

}

