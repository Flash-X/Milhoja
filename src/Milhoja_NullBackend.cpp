#include "Milhoja_NullBackend.h"

#include <cstdlib>
#include <cstring>

namespace milhoja {

/**
 * Refer to the RuntimeBackend documentation for more information.
 *
 * @todo We should implement a generic thread-safe class that manages a CPU
 * memory pool.  Hopefully, all backends should be able to use that single
 * class.  Update this and the paired release function once done.
 */
void      NullBackend::requestCpuMemory(const std::size_t nBytes,
                                        void** ptr) {
    if (!ptr) {
        std::string  errMsg = "[NullBackend::requestCpuMemory] ";
        errMsg += "Null handle given\n";
        throw std::invalid_argument(errMsg);
    } else if (*ptr) {
        std::string  errMsg = "[NullBackend::requestCpuMemory] ";
        errMsg += "Internal pointer already set\n";
        throw std::invalid_argument(errMsg);
    } else if (nBytes == 0) {
        std::string  errMsg = "[NullBackend::requestCpuMemory] ";
        errMsg += "Requests of zero indicate logical error\n";
        throw std::invalid_argument(errMsg);
    }

    *ptr = std::malloc(nBytes);
    std::memset(*ptr, 0, nBytes);
}

/**
 * Refer to the RuntimeBackend documentation for more information.
 */
void      NullBackend::releaseCpuMemory(void** ptr) {
    if (!ptr) {
        std::string  errMsg = "[NullBackend::releaseCpuMemory] ";
        errMsg += "Null handle given\n";
        throw std::invalid_argument(errMsg);
    } else if (!*ptr) {
        std::string  errMsg = "[NullBackend::releaseCpuMemory] ";
        errMsg += "Internal pointer is null\n";
        throw std::invalid_argument(errMsg);
    }

    std::free(*ptr);
    *ptr = nullptr;
}

}

