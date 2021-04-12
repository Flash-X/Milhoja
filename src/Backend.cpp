#include "Backend.h"

#include <stdexcept>

#ifdef USE_CUDA_BACKEND
#include "CudaBackend.h"
#else
#include "NullBackend.h"
#endif

#include "OrchestrationLogger.h"

namespace orchestration {

bool           Backend::instantiated_ = false;
unsigned int   Backend::nStreams_ = 0;
std::size_t    Backend::nBytesInMemoryPools_ = 0;

/**
 * Provide the Singleton design pattern with the runtime parameters needed to
 * appropriately instantiate and initialize the desired concrete Backend class.
 * This function must be called before using instance() to gain access to the
 * Singleton object.
 *
 * @param nStreams - the maximum number of streams that the runtime is allowed
 * to use at any point in time.
 * @param nBytesInMemoryPools - the amount of memory to allocate in memory
 * pools.  Refer to the documentation for each backend to determine what memory
 * pools will be allocated.
 */
void   Backend::instantiate(const unsigned int nStreams,
                            const std::size_t  nBytesInMemoryPools) {
    Logger::instance().log("[Backend] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[Backend::instantiate] Already instantiated");
    }

    // The responsibility of error checking these falls on the code that
    // consumes them later.
    nStreams_ = nStreams;
    nBytesInMemoryPools_ = nBytesInMemoryPools;

    instantiated_ = true;

    // Create/initialize backend
    instance();

    Logger::instance().log("[Backend] Created and ready for use");
}

/**
 * Obtain access to the Backend Singleton object.
 *
 * @return A reference to the object
 */
Backend&   Backend::instance(void) {
    if(!instantiated_) {
        throw std::logic_error("[Backend::instance] instantiate first");
    }

#ifdef USE_CUDA_BACKEND
    static CudaBackend singleton{nStreams_, nBytesInMemoryPools_};
#else
    static NullBackend singleton{};
    Logger::instance().log("[Backend] No runtime backend needed");
#endif

    return singleton;
}

}

