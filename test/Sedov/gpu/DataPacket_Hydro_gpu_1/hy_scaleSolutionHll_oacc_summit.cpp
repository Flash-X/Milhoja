#include "Hydro.h"

#include <Milhoja.h>

#include "Sedov.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::scaleSolutionHll_oacc_summit(const milhoja::IntVect* lo_d,
                                      const milhoja::IntVect* hi_d,
                                      const milhoja::FArray4D* Uin_d,
                                      milhoja::FArray4D* Uout_d) {
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
                Uout_d->at(i, j, k, VELX_VAR) =   Uin_d->at(i, j, k, VELX_VAR)
                                                * Uin_d->at(i, j, k, DENS_VAR);
                Uout_d->at(i, j, k, VELY_VAR) =   Uin_d->at(i, j, k, VELY_VAR)
                                                * Uin_d->at(i, j, k, DENS_VAR);
                Uout_d->at(i, j, k, VELZ_VAR) =   Uin_d->at(i, j, k, VELZ_VAR)
                                                * Uin_d->at(i, j, k, DENS_VAR);
                Uout_d->at(i, j, k, ENER_VAR) =   Uin_d->at(i, j, k, ENER_VAR)
                                                * Uin_d->at(i, j, k, DENS_VAR);
                Uout_d->at(i, j, k, DENS_VAR) =   Uin_d->at(i, j, k, DENS_VAR);
            }
        }
    }
}

