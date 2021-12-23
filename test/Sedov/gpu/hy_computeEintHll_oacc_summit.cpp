#include "Hydro.h"

#include "Sedov.h"

#ifndef MILHOJA_ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::computeEintHll_oacc_summit(const milhoja::IntVect* lo_d,
                                    const milhoja::IntVect* hi_d,
                                    milhoja::FArray4D* U_d) {
    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    // Correct energy
    milhoja::Real   norm2_sqr = 0.0_wp;

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e; ++k) {
        for     (int j=j_s; j<=j_e; ++j) {
            for (int i=i_s; i<=i_e; ++i) {
                norm2_sqr =   U_d->at(i, j, k, VELX_VAR) * U_d->at(i, j, k, VELX_VAR)
                            + U_d->at(i, j, k, VELY_VAR) * U_d->at(i, j, k, VELY_VAR)
                            + U_d->at(i, j, k, VELZ_VAR) * U_d->at(i, j, k, VELZ_VAR);
                U_d->at(i, j, k, EINT_VAR) =    U_d->at(i, j, k, ENER_VAR)
                                             - (0.5_wp * norm2_sqr);
            }
        }
    }
}

