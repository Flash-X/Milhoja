#ifndef DRIVER_H__
#define DRIVER_H__

#include <Milhoja_real.h>

namespace Driver {
    // FIXME: Needed by DataPackets at the moment so that dt can be copied to
    // GPU memory by each data packet.  Value is not important.
    constexpr milhoja::Real     dt = 0.0;
};

#endif

