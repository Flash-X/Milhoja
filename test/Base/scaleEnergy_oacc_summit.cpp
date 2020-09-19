#include "scaleEnergy.h"

void StaticPhysicsRoutines::scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                    const orchestration::IntVect* hi_d,
                                                    const orchestration::FArray1D* xCoords_d,
                                                    const orchestration::FArray1D* yCoords_d,
                                                    orchestration::FArray4D* f_d,
                                                    const orchestration::Real scaleFactor,
                                                    const int streamId_h) {
    orchestration::Real    x = 0.0;
    orchestration::Real    y = 0.0;

    #pragma acc data create(x, y) \
                     deviceptr(lo_d, hi_d, xCoords_d, yCoords_d, f_d)
    {
        #pragma acc parallel loop default(none) async(streamId_h)
        for         (int k=lo_d->K(); k<=hi_d->K(); ++k) {
            #pragma acc loop
            for     (int j=lo_d->J(); j<=hi_d->J(); ++j) {
                #pragma acc loop
                for (int i=lo_d->I(); i<=hi_d->I(); ++i) {
                    x = xCoords_d->at(i);
                    y = yCoords_d->at(j);
                    f_d->at(i, j, k, ENER_VAR_C) = 5.0*x*y;
                }
            }
        }
        #pragma acc wait(streamId_h)
    }
}

