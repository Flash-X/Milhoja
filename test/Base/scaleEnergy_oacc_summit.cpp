#include "scaleEnergy.h"

void StaticPhysicsRoutines::scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                    const orchestration::IntVect* hi_d,
                                                    const orchestration::FArray1D* xCoords_d,
                                                    const orchestration::FArray1D* yCoords_d,
                                                    orchestration::FArray4D* U_d,
                                                    const orchestration::Real scaleFactor,
                                                    const int queue_h) {
    #pragma acc data deviceptr(lo_d, hi_d, xCoords_d, yCoords_d, U_d)
    {
        #pragma acc parallel default(none) async(queue_h)
        {
            int                    i_s = lo_d->I();
            int                    j_s = lo_d->J();
            int                    i_e = hi_d->I();
            int                    j_e = hi_d->J();
            orchestration::Real    x = 0.0;
            orchestration::Real    y = 0.0;
            // THe OFFLINE TOOLCHAIN should realize that there is no need for
            // the k loop and eliminate it to see if this helps performance on
            // the GPU.
            #pragma acc loop collapse(2)
            for     (int j=j_s; j<=j_e; ++j) {
                for (int i=i_s; i<=i_e; ++i) {
                    x = xCoords_d->at(i);
                    y = yCoords_d->at(j);
                    U_d->at(i, j, 0, ENER_VAR_C) *= scaleFactor*x*y;
                }
            }
        }
        #pragma acc wait(queue_h)
    }
}

