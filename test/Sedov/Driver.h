#ifndef DRIVER_H__
#define DRIVER_H__

#include <Milhoja_real.h>

namespace Driver {
    // This is new as dt needs to be public so that action routines can grab
    // this value and pass into the PUD action routines.
    // TODO: Include a pointer to the variable (i.e. dt_d) that stores dt in GPU
    // device memory.
    extern milhoja::Real     dt;
    extern milhoja::Real     simTime;

    // Each tests needs to define its own version of this function.
    void  executeSimulation(void);
};

#endif

