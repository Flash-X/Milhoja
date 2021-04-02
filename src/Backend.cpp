#include "Backend.h"

#include <stdexcept>

#include "OrchestrationLogger.h"

#ifdef USE_CUDA_BACKEND
#include "CudaBackend.h"
#else
#include "NullBackend.h"
#endif

namespace orchestration {

bool           Backend::instantiated_ = false;
unsigned int   Backend::nStreams_ = 0;
std::size_t    Backend::nBytesInMemoryPools_ = 0;

/**
 * 
 *
 * \return 
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
 * instace gets a reference to the singleton Backend object.
 *
 * @return A reference to the singleton object, which has been downcast
 *         to a Backend type
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

