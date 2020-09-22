#include "computeLaplacianDensity.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(const orchestration::IntVect* lo_d,
                                                                const orchestration::IntVect* hi_d,
                                                                orchestration::FArray4D* f_d,
                                                                orchestration::FArray4D* scratch_d,
                                                                const orchestration::RealVect* deltas_d,
                                                                const int streamId_h) {
    using namespace orchestration;

    #pragma acc data deviceptr(deltas_d, lo_d, hi_d, f_d, scratch_d)
    {
        #pragma acc parallel default(none) async(streamId_h)
        {
            int     i_s = lo_d->I();
            int     j_s = lo_d->J();
            int     i_e = hi_d->I();
            int     j_e = hi_d->J();
            Real    dx_sqr_inv = 1.0 / (deltas_d->I() * deltas_d->I());
            Real    dy_sqr_inv = 1.0 / (deltas_d->J() * deltas_d->J());

            // Compute Laplacian in scratch
            // THe OFFLINE TOOLCHAIN should realize that there is no need for
            // the k loop and eliminate it to see if this helps performance on
            // the GPU.
            #pragma acc loop collapse(2)
            for     (int j=j_s; j<=j_e; ++j) {
                for (int i=i_s; i<=i_e; ++i) {
                      scratch_d->at(i, j, 0, 0) = 
                               (     (  f_d->at(i-1, j,   0, DENS_VAR_C)
                                      + f_d->at(i+1, j,   0, DENS_VAR_C))
                                - 2.0 * f_d->at(i,   j,   0, DENS_VAR_C) ) * dx_sqr_inv
                             + (     (  f_d->at(i  , j-1, 0, DENS_VAR_C)
                                      + f_d->at(i  , j+1, 0, DENS_VAR_C))
                                - 2.0 * f_d->at(i,   j,   0, DENS_VAR_C) ) * dy_sqr_inv;
                 }
            }
        }
        #pragma acc wait(streamId_h)
    }
}

