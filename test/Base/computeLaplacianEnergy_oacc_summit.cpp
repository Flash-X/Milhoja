#include "computeLaplacianEnergy.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(const orchestration::IntVect* lo_d,
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
            #pragma acc loop
            for     (int j=j_s; j<=j_e; ++j) {
                #pragma acc loop
                for (int i=i_s; i<=i_e; ++i) {
                      scratch_d->at(i, j, 0, 0) = 
                               (     (  f_d->at(i-1, j,   0, ENER_VAR_C)
                                      + f_d->at(i+1, j,   0, ENER_VAR_C))
                                - 2.0 * f_d->at(i,   j,   0, ENER_VAR_C) ) * dx_sqr_inv
                             + (     (  f_d->at(i  , j-1, 0, ENER_VAR_C)
                                      + f_d->at(i  , j+1, 0, ENER_VAR_C))
                                - 2.0 * f_d->at(i,   j,   0, ENER_VAR_C) ) * dy_sqr_inv;
                 }
            }
            

            // Overwrite interior of given block with Laplacian result
            // TODO: In the case of a data packet, we could have the input data given as
            // a pointer to CC1 and directly write the result to CC2.  When copying the
            // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
            // would be unnecessary.
            #pragma acc loop
            for     (int j=j_s; j<=j_e; ++j) {
                #pragma acc loop
                for (int i=i_s; i<=i_e; ++i) {
                    f_d->at(i, j, 0, ENER_VAR_C) = scratch_d->at(i, j, 0, 0);
                 }
            } 
        }
        #pragma acc wait(streamId_h)
    }
}

