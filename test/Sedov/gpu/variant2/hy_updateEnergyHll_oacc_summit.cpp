#include "Hydro.h"

#include <Milhoja.h>

#include "Sedov.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::updateEnergyHll_oacc_summit(const milhoja::IntVect* lo_d,
                                     const milhoja::IntVect* hi_d,
                                     const milhoja::FArray4D* Uin_d,
                                     milhoja::FArray4D* Uout_d,
                                     const milhoja::FArray4D* flX_d,
                                     const milhoja::FArray4D* flY_d,
                                     const milhoja::FArray4D* flZ_d) {
    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    milhoja::Real    invNewDens = 0.0_wp;

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e; ++k) {
        for     (int j=j_s; j<=j_e; ++j) {
            for (int i=i_s; i<=i_e; ++i) {
                invNewDens = 1.0_wp / Uout_d->at(i, j, k, DENS_VAR);

#if MILHOJA_NDIM == 1
                Uout_d->at(i, j, k, ENER_VAR) = (  Uin_d->at(i,   j, k, ENER_VAR) * Uin_d->at(i, j, k, DENS_VAR)
                                                 + flX_d->at(i,   j, k, HY_ENER_FLUX)
                                                 - flX_d->at(i+1, j, k, HY_ENER_FLUX) ) * invNewDens;
#elif MILHOJA_NDIM == 2
                Uout_d->at(i, j, k, ENER_VAR) = (  Uin_d->at(i,   j,   k, ENER_VAR) * Uin_d->at(i, j, k, DENS_VAR)
                                                 + flX_d->at(i,   j,   k, HY_ENER_FLUX)
                                                 - flX_d->at(i+1, j,   k, HY_ENER_FLUX)
                                                 + flY_d->at(i,   j,   k, HY_ENER_FLUX)
                                                 - flY_d->at(i,   j+1, k, HY_ENER_FLUX) ) * invNewDens;
#elif MILHOJA_NDIM == 3
                Uout_d->at(i, j, k, ENER_VAR) = (  Uin_d->at(i,   j,   k,   ENER_VAR) * Uin_d->at(i, j, k, DENS_VAR)
                                                 + flX_d->at(i,   j,   k,   HY_ENER_FLUX)
                                                 - flX_d->at(i+1, j,   k,   HY_ENER_FLUX)
                                                 + flY_d->at(i,   j,   k,   HY_ENER_FLUX)
                                                 - flY_d->at(i,   j+1, k,   HY_ENER_FLUX)
                                                 + flZ_d->at(i,   j,   k,   HY_ENER_FLUX)
                                                 - flZ_d->at(i,   j,   k+1, HY_ENER_FLUX) ) * invNewDens;
#endif
            }
        }
    }
}

