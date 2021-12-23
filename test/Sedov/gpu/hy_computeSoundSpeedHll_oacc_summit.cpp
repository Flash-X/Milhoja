#include "Hydro.h"

#include <cmath>

#include <Milhoja.h>

#include "Sedov.h"

#ifndef MILHOJA_ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::computeSoundSpeedHll_oacc_summit(const milhoja::IntVect* lo_d,
                                          const milhoja::IntVect* hi_d,
                                          const milhoja::FArray4D* U_d,
                                          milhoja::FArray4D* auxC_d) {
    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s-MILHOJA_K3D; k<=k_e+MILHOJA_K3D; ++k) {
        for     (int j=j_s-MILHOJA_K2D; j<=j_e+MILHOJA_K2D; ++j) {
            for (int i=i_s-MILHOJA_K1D; i<=i_e+MILHOJA_K1D; ++i) {
                auxC_d->at(i, j, k, 0) = sqrt(  U_d->at(i, j, k, GAMC_VAR)
                                              * U_d->at(i, j, k, PRES_VAR)
                                              / U_d->at(i, j, k, DENS_VAR) );
            }
        }
    }
}

