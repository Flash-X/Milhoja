#ifndef USE_OPENACC
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianEnergy.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                                               const orchestration::IntVect* hi_d,
                                                               orchestration::FArray4D* Uin_d,
                                                               orchestration::FArray4D* Uout_d,
                                                               const orchestration::RealVect* deltas_d,
                                                               const int queue_h) {
    using namespace orchestration;

    #pragma acc data deviceptr(deltas_d, lo_d, hi_d, Uin_d, Uout_d)
    {
        #pragma acc parallel default(none) async(queue_h)
        {
            int     i_s = lo_d->I();
            int     j_s = lo_d->J();
            int     i_e = hi_d->I();
            int     j_e = hi_d->J();
            Real    dx_sqr_inv = 1.0 / (deltas_d->I() * deltas_d->I());
            Real    dy_sqr_inv = 1.0 / (deltas_d->J() * deltas_d->J());

            // Compute Laplacian in Uout
            // THe OFFLINE TOOLCHAIN should realize that there is no need for
            // the k loop and eliminate it to see if this helps performance on
            // the GPU.
            #pragma acc loop collapse(2)
            for     (int j=j_s; j<=j_e; ++j) {
                for (int i=i_s; i<=i_e; ++i) {
                      Uout_d->at(i, j, 0, ENER_VAR_C) = 
                               (     (  Uin_d->at(i-1, j,   0, ENER_VAR_C)
                                      + Uin_d->at(i+1, j,   0, ENER_VAR_C))
                                - 2.0 * Uin_d->at(i,   j,   0, ENER_VAR_C) ) * dx_sqr_inv
                             + (     (  Uin_d->at(i  , j-1, 0, ENER_VAR_C)
                                      + Uin_d->at(i  , j+1, 0, ENER_VAR_C))
                                - 2.0 * Uin_d->at(i,   j,   0, ENER_VAR_C) ) * dy_sqr_inv;
                 }
            }
            // The OFFLINE TOOLCHAIN should figure out that Uin/Uout can be used
            // here to remove the copyback.
        }
        #pragma acc wait(queue_h)
    }
}

