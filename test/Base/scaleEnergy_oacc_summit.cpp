#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "scaleEnergy.h"

void StaticPhysicsRoutines::scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                    const orchestration::IntVect* hi_d,
                                                    const orchestration::FArray1D* xCoords_d,
                                                    const orchestration::FArray1D* yCoords_d,
                                                    orchestration::FArray4D* U_d,
                                                    const orchestration::Real scaleFactor) {
    int                    i_s = lo_d->I();
    int                    j_s = lo_d->J();
    int                    i_e = hi_d->I();
    int                    j_e = hi_d->J();
    orchestration::Real    x = 0.0;
    orchestration::Real    y = 0.0;
    // THe OFFLINE TOOLCHAIN should realize that there is no need for
    // the k loop and eliminate it to see if this helps performance on
    // the GPU.
    #pragma acc loop vector collapse(2)
    for     (int j=j_s; j<=j_e; ++j) {
        for (int i=i_s; i<=i_e; ++i) {
            x = xCoords_d->at(i);
            y = yCoords_d->at(j);
            U_d->at(i, j, 0, ENER_VAR_C) *= scaleFactor*x*y;
        }
    }
}

