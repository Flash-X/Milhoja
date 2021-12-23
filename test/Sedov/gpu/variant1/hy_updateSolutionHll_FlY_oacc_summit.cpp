#include "Hydro.h"

#include "Sedov.h"

#ifndef MILHOJA_ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::updateSolutionHll_FlY_oacc_summit(const milhoja::IntVect* lo_d,
                                           const milhoja::IntVect* hi_d,
                                           milhoja::FArray4D* U_d,
                                           const milhoja::FArray4D* flY_d) {
    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e; ++k) {
        for     (int j=j_s; j<=j_e; ++j) {
            for (int i=i_s; i<=i_e; ++i) {
                U_d->at(i, j, k, DENS_VAR) += (  flY_d->at(i, j,   k, HY_DENS_FLUX)
                                               - flY_d->at(i, j+1, k, HY_DENS_FLUX) );
                U_d->at(i, j, k, VELX_VAR) += (  flY_d->at(i, j,   k, HY_XMOM_FLUX)
                                               - flY_d->at(i, j+1, k, HY_XMOM_FLUX) );
                U_d->at(i, j, k, VELY_VAR) += (  flY_d->at(i, j,   k, HY_YMOM_FLUX)
                                               - flY_d->at(i, j+1, k, HY_YMOM_FLUX) );
                U_d->at(i, j, k, VELZ_VAR) += (  flY_d->at(i, j,   k, HY_ZMOM_FLUX)
                                               - flY_d->at(i, j+1, k, HY_ZMOM_FLUX) );
                U_d->at(i, j, k, ENER_VAR) += (  flY_d->at(i, j,   k, HY_ENER_FLUX)
                                               - flY_d->at(i, j+1, k, HY_ENER_FLUX) );
            }
        }
    }
}

