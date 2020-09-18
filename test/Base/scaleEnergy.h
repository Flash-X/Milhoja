#ifndef SCALE_ENERGY_H__
#define SCALE_ENERGY_H__

#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"

namespace ThreadRoutines {
    // OFFLINE TOOLCHAIN:  Should streamId be excluded from the base declaration
    //                     and definition with the understanding that the
    //                     toolchain would add it when it translates such
    //                     routines to the final version using the desired 
    //                     directive (e.g. OpenACC or OpenMP)?
    // OFFLINE TOOLCHAIN:  Could these be defined with references with
    //                     conversion to pointers as needed by the offline
    //                     toolchain when generating the final version using
    //                     the desired directive (e.g. OpenACC or OpenMP)?
    void scaleEnergy(const orchestration::IntVect& lo,
                     const orchestration::IntVect& hi,
                     const orchestration::FArray1D& xCoords,
                     const orchestration::FArray1D& yCoords,
                     orchestration::FArray4D& f,
                     const orchestration::Real scaleFactor);
}

#endif

