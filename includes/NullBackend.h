#ifndef NULL_BACKEND_H__
#define NULL_BACKEND_H__

#include "Backend.h"

namespace orchestration {

class NullBackend : public Backend {
public:
    ~NullBackend(void)       {};

    NullBackend(NullBackend&)                  = delete;
    NullBackend(const NullBackend&)            = delete;
    NullBackend(NullBackend&&)                 = delete;
    NullBackend& operator=(NullBackend&)       = delete;
    NullBackend& operator=(const NullBackend&) = delete;
    NullBackend& operator=(NullBackend&&)      = delete;

private:
    NullBackend(void)        {};

    // Needed for polymorphic singleton
    friend Backend&   Backend::instance();
};

}

#endif

