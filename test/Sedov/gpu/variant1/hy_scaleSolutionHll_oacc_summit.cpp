#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include "Flash.h"

void hy::scaleSolutionHll_oacc_summit(const orchestration::IntVect* lo_d,
                                      const orchestration::IntVect* hi_d,
                                      const orchestration::FArray4D* Uin_d,
                                      orchestration::FArray4D* Uout_d) {
    using namespace orchestration;

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

