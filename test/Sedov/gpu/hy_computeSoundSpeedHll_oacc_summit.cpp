#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include <cmath>

#include "Flash.h"

void hy::computeSoundSpeedHll_oacc_summit(const orchestration::IntVect* lo_d,
                                          const orchestration::IntVect* hi_d,
                                          const orchestration::FArray4D* U_d,
                                          orchestration::FArray4D* auxC_d) {
    using namespace orchestration;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s-K3D; k<=k_e+K3D; ++k) {
        for     (int j=j_s-K2D; j<=j_e+K2D; ++j) {
            for (int i=i_s-K1D; i<=i_e+K1D; ++i) {
                auxC_d->at(i, j, k, 0) = sqrt(  U_d->at(i, j, k, GAMC_VAR)
                                              * U_d->at(i, j, k, PRES_VAR)
                                              / U_d->at(i, j, k, DENS_VAR) );
            }
        }
    }
}

