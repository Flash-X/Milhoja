#ifndef DRIVER_H__
#define DRIVER_H__

#include "Grid_REAL.h"

namespace Driver {
    // This is new as dt needs to be public so that action routines can grab
    // this value and pass into the PUD action routines.
    // TODO: Include a pointer to the variable (i.e. dt_d) that stores dt in GPU
    // device memory.
    extern orchestration::Real     dt;
    extern orchestration::Real     simTime;
};

namespace dr {
    // This is how I fake a Driver_computeDt
    constexpr orchestration::Real   dtAfter          = 5.0e-5_wp;

    // Fix runtime parameters
    constexpr unsigned int          writeEveryNSteps = 10;
};

#endif

