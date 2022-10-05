#include "computeLaplacianEnergy.h"

#include <Milhoja.h>

#include "Base.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(const milhoja::IntVect* lo_d,
                                                               const milhoja::IntVect* hi_d,
                                                               const milhoja::FArray4D* Uin_d,
                                                               milhoja::FArray4D* Uout_d,
                                                               const milhoja::RealVect* deltas_d) {
    int              i_s = lo_d->I();
    int              j_s = lo_d->J();
    int              i_e = hi_d->I();
    int              j_e = hi_d->J();
    milhoja::Real    dx_sqr_inv = 1.0 / (deltas_d->I() * deltas_d->I());
    milhoja::Real    dy_sqr_inv = 1.0 / (deltas_d->J() * deltas_d->J());

    // Compute Laplacian in Uout
    // THe OFFLINE TOOLCHAIN should realize that there is no need for
    // the k loop and eliminate it to see if this helps performance on
    // the GPU.
    #pragma acc loop vector collapse(2)
    for     (int j=j_s; j<=j_e; ++j) {
        for (int i=i_s; i<=i_e; ++i) {
              Uout_d->at(i, j, 0, ENER_VAR) = 
                       (     (  Uin_d->at(i-1, j,   0, ENER_VAR)
                              + Uin_d->at(i+1, j,   0, ENER_VAR))
                        - 2.0 * Uin_d->at(i,   j,   0, ENER_VAR) ) * dx_sqr_inv
                     + (     (  Uin_d->at(i  , j-1, 0, ENER_VAR)
                              + Uin_d->at(i  , j+1, 0, ENER_VAR))
                        - 2.0 * Uin_d->at(i,   j,   0, ENER_VAR) ) * dy_sqr_inv;
         }
    }
    // The OFFLINE TOOLCHAIN should figure out that Uin/Uout can be used
    // here to remove the copyback.
}

