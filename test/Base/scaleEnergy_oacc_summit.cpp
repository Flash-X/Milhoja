#include "scaleEnergy.h"

void StaticPhysicsRoutines::scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                    const orchestration::IntVect* hi_d,
                                                    const orchestration::Real* xCoords_d,
                                                    const orchestration::Real* yCoords_d,
                                                    orchestration::FArray4D* f_d,
                                                    const orchestration::Real scaleFactor,
                                                    const int streamId_h) {
    #pragma acc data deviceptr(lo_d, hi_d, xCoords_d, yCoords_d, f_d)
    {
        #pragma acc parallel default(none) async(streamId_h)
        {
            int                    i_s = lo_d->I();
            int                    j_s = lo_d->J();
            int                    k_s = lo_d->K();
            int                    i_e = hi_d->I();
            int                    j_e = hi_d->J();
            int                    k_e = hi_d->K();
            orchestration::Real    x = 0.0;
            orchestration::Real    y = 0.0;
            #pragma acc loop
            for         (int k=k_s; k<=k_e; ++k) {
                #pragma acc loop
                for     (int j=j_s; j<=j_e; ++j) {
                    y = yCoords_d[j - j_s];
                    #pragma acc loop
                    for (int i=i_s; i<=i_e; ++i) {
                        x = xCoords_d[i - i_s];
                        f_d->at(i, j, k, ENER_VAR_C) *= scaleFactor*x*y;
                    }
                }
            }
        }
        #pragma acc wait(streamId_h)
    }
}

