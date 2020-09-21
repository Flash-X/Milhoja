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
            Real    dx_sqr_inv = 1.0 / (deltas_d->I() * deltas_d->I());
            Real    dy_sqr_inv = 1.0 / (deltas_d->J() * deltas_d->J());

            // Compute Laplacian in scratch
            #pragma acc loop
            for         (int k=lo_d->K(); k<=hi_d->K(); ++k) {
                #pragma acc loop
                for     (int j=lo_d->J(); j<=hi_d->J(); ++j) {
                    #pragma acc loop
                    for (int i=lo_d->I(); i<=hi_d->I(); ++i) {
                          scratch_d->at(i, j, k, 0) = 
                                   (     (  f_d->at(i-1, j,   k, DENS_VAR_C)
                                          + f_d->at(i+1, j,   k, DENS_VAR_C))
                                    - 2.0 * f_d->at(i,   j,   k, DENS_VAR_C) ) * dx_sqr_inv
                                 + (     (  f_d->at(i  , j-1, k, DENS_VAR_C)
                                          + f_d->at(i  , j+1, k, DENS_VAR_C))
                                    - 2.0 * f_d->at(i,   j,   k, DENS_VAR_C) ) * dy_sqr_inv;
                     }
                }
            }

            // Overwrite interior of given block with Laplacian result
            // TODO: In the case of a data packet, we could have the input data given as
            // a pointer to CC1 and directly write the result to CC2.  When copying the
            // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
            // would be unnecessary.
            #pragma acc loop
            for         (int k=lo_d->K(); k<=hi_d->K(); ++k) {
                #pragma acc loop
                for     (int j=lo_d->J(); j<=hi_d->J(); ++j) {
                    #pragma acc loop
                    for (int i=lo_d->I(); i<=hi_d->I(); ++i) {
                        f_d->at(i, j, k, DENS_VAR_C) = scratch_d->at(i, j, k, 0);
                     }
                } 
            }
        }
        #pragma acc wait(streamId_h)
    }
}

