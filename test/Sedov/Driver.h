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

#endif

