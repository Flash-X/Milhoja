#ifndef DRIVER_H__
#define DRIVER_H__

#include "Grid_REAL.h"

namespace Driver {
    // FIXME: Needed by CudaDataPacket at the moment so that dt can be copied to
    // GPU memory by each data packet.  Value is not important.
    constexpr orchestration::Real     dt = 0.0;
};

#endif

