#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianFused.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianFusedKernels_oacc_summit(const orchestration::IntVect* lo_d,
                                                                     const orchestration::IntVect* hi_d,
                                                                     orchestration::FArray4D* Uin_d,
                                                                     orchestration::FArray4D* Uout_d,
                                                                     const orchestration::RealVect* deltas_d) {
    using namespace orchestration;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    Real    dx_sqr_inv = 1.0 / (deltas_d->I() * deltas_d->I());
    Real    dy_sqr_inv = 1.0 / (deltas_d->J() * deltas_d->J());

    // Compute Laplacian in Uout
    // The OFFLINE TOOLCHAIN should realize that there is no need for
    // the k loop and eliminate it to see if this helps performance on
    // the GPU.
    // The OFFLINE TOOLCHAIN determined that it could fuse the two independent
    // Laplacian computations
    #pragma acc loop vector collapse(2)
    for     (int j=j_s; j<=j_e; ++j) {
        for (int i=i_s; i<=i_e; ++i) {
              Uout_d->at(i, j, 0, DENS_VAR_C) = 
                       (     (  Uin_d->at(i-1, j,   0, DENS_VAR_C)
                              + Uin_d->at(i+1, j,   0, DENS_VAR_C))
                        - 2.0 * Uin_d->at(i,   j,   0, DENS_VAR_C) ) * dx_sqr_inv
                     + (     (  Uin_d->at(i  , j-1, 0, DENS_VAR_C)
                              + Uin_d->at(i  , j+1, 0, DENS_VAR_C))
                        - 2.0 * Uin_d->at(i,   j,   0, DENS_VAR_C) ) * dy_sqr_inv;

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

