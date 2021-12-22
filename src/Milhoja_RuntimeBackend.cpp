#include "Milhoja_RuntimeBackend.h"

#include <stdexcept>

#include "Milhoja_Logger.h"
#ifdef USE_CUDA_BACKEND
#include "Milhoja_CudaBackend.h"
#else
#include "Milhoja_NullBackend.h"
#endif

namespace milhoja {

bool           RuntimeBackend::instantiated_ = false;
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
void   RuntimeBackend::instantiate(const unsigned int nStreams,
                            const std::size_t  nBytesInMemoryPools) {
    Logger::instance().log("[RuntimeBackend] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[RuntimeBackend::instantiate] Already instantiated");
    }

    // The responsibility of error checking these falls on the code that
    // consumes them later.
    nStreams_ = nStreams;
    nBytesInMemoryPools_ = nBytesInMemoryPools;

    instantiated_ = true;

    // Create/initialize backend
    instance();

    Logger::instance().log("[RuntimeBackend] Created and ready for use");
}

/**
 * Obtain access to the RuntimeBackend Singleton object.
 */
RuntimeBackend&   RuntimeBackend::instance(void) {
    if(!instantiated_) {
        throw std::logic_error("[RuntimeBackend::instance] instantiate first");
    }

#ifdef USE_CUDA_BACKEND
    static CudaBackend singleton{nStreams_, nBytesInMemoryPools_};
#else
    static NullBackend singleton{};
    Logger::instance().log("[RuntimeBackend] No runtime backend needed");
#endif

    return singleton;
}

RuntimeBackend::~RuntimeBackend(void) {
    instantiated_ = false;
    nStreams_ = 0;
    nBytesInMemoryPools_ = 0;
};

}

