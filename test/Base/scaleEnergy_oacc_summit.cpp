#include "scaleEnergy.h"

void StaticPhysicsRoutines::scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                    const orchestration::IntVect* hi_d,
                                                    const orchestration::Real* xCoords_h,
                                                    const orchestration::Real* yCoords_h,
                                                    orchestration::FArray4D* f_d,
                                                    const orchestration::Real scaleFactor,
                                                    const int streamId_h) {
    int                    i_s = 0;
    int                    j_s = 0;
    int                    k_s = 0;
    int                    i_e = 0;
    int                    j_e = 0;
    int                    k_e = 0;
    orchestration::Real    x = 0.0;
    orchestration::Real    y = 0.0;

    #pragma acc data create(i_s, j_s, k_s, i_e, j_e, k_e, x, y) \
                     copyin(xCoords_h[0:NXB], yCoords_h[0:NYB]) \
                     deviceptr(lo_d, hi_d, f_d)
    {
        #pragma acc kernels default(none) async(streamId_h)
        {
            i_s = lo_d->I();
            j_s = lo_d->J();
            k_s = lo_d->K();
            i_e = hi_d->I();
            j_e = hi_d->J();
            k_e = hi_d->K();
            for         (int k=k_s; k<=k_e; ++k) {
                for     (int j=j_s; j<=j_e; ++j) {
                    y = yCoords_h[j - j_s];
                    for (int i=i_s; i<=i_e; ++i) {
                        x = xCoords_h[i - i_s];
                        f_d->at(i, j, k, ENER_VAR_C) *= scaleFactor*x*y;
                    }
                }
            }
        }
        #pragma acc wait(streamId_h)
    }
}

