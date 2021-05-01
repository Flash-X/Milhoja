#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include "Flash.h"

void hy::updateSolutionHll_oacc_summit(const orchestration::IntVect* lo_d,
                                       const orchestration::IntVect* hi_d,
                                       const orchestration::FArray4D* Uin_d,
                                       orchestration::FArray4D* Uout_d,
                                       const orchestration::FArray4D* flX_d,
                                       const orchestration::FArray4D* flY_d,
                                       const orchestration::FArray4D* flZ_d) {
    using namespace orchestration;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

#ifdef EINT_VAR_C
    Real    norm2_sqr = 0.0_wp;
#endif
    Real    densOld = 0.0_wp;
    Real    densNew = 0.0_wp;
    Real    densNew_inv = 0.0_wp;
    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e; ++k) {
        for     (int j=j_s; j<=j_e; ++j) {
            for (int i=i_s; i<=i_e; ++i) {
                // Update density first
                densOld = Uin_d->at(i, j, k, DENS_VAR_C);
#if NDIM == 1
                densNew =   densOld
                          + flX_d->at(i,   j, k, HY_DENS_FLUX_C)
                          - flX_d->at(i+1, j, k, HY_DENS_FLUX_C);
#elif NDIM == 2
                densNew =   densOld
                          + flX_d->at(i,   j,   k, HY_DENS_FLUX_C)
                          - flX_d->at(i+1, j,   k, HY_DENS_FLUX_C)
                          + flY_d->at(i,   j,   k, HY_DENS_FLUX_C)
                          - flY_d->at(i,   j+1, k, HY_DENS_FLUX_C);
#elif NDIM == 3
                densNew =   densOld 
                          + flX_d->at(i,   j,   k,   HY_DENS_FLUX_C)
                          - flX_d->at(i+1, j,   k,   HY_DENS_FLUX_C)
                          + flY_d->at(i,   j,   k,   HY_DENS_FLUX_C)
                          - flY_d->at(i,   j+1, k,   HY_DENS_FLUX_C)
                          + flZ_d->at(i,   j,   k,   HY_DENS_FLUX_C)
                          - flZ_d->at(i,   j,   k+1, HY_DENS_FLUX_C);
#endif
                Uout_d->at(i, j, k, DENS_VAR_C) = densNew;
                densNew_inv = 1.0_wp / densNew;

                // velocities and total energy can be updated independently
                // using density result
#if NDIM == 1
                Uout_d->at(i, j, k, VELX_VAR_C) = (    Uin_d->at(i,   j, k, VELX_VAR_C) * densOld
                                                     + flX_d->at(i,   j, k, HY_XMOM_FLUX_C)
                                                     - flX_d->at(i+1, j, k, HY_XMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELY_VAR_C) = (    Uin_d->at(i,   j, k, VELY_VAR_C) * densOld
                                                     + flX_d->at(i,   j, k, HY_YMOM_FLUX_C)
                                                     - flX_d->at(i+1, j, k, HY_YMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELZ_VAR_C) = (    Uin_d->at(i,   j, k, VELZ_VAR_C) * densOld
                                                     + flX_d->at(i,   j, k, HY_ZMOM_FLUX_C)
                                                     - flX_d->at(i+1, j, k, HY_ZMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, ENER_VAR_C) = (    Uin_d->at(i,   j, k, ENER_VAR_C) * densOld
                                                     + flX_d->at(i,   j, k, HY_ENER_FLUX_C)
                                                     - flX_d->at(i+1, j, k, HY_ENER_FLUX_C) ) * densNew_inv;
#elif NDIM == 2
                Uout_d->at(i, j, k, VELX_VAR_C) = (    Uin_d->at(i,   j,   k, VELX_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k, HY_XMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k, HY_XMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k, HY_XMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k, HY_XMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELY_VAR_C) = (    Uin_d->at(i,   j,   k, VELY_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k, HY_YMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k, HY_YMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k, HY_YMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k, HY_YMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELZ_VAR_C) = (    Uin_d->at(i,   j,   k, VELZ_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k, HY_ZMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k, HY_ZMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k, HY_ZMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k, HY_ZMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, ENER_VAR_C) = (    Uin_d->at(i,   j,   k, ENER_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k, HY_ENER_FLUX_C)
                                                     - flX_d->at(i+1, j,   k, HY_ENER_FLUX_C)
                                                     + flY_d->at(i,   j,   k, HY_ENER_FLUX_C)
                                                     - flY_d->at(i,   j+1, k, HY_ENER_FLUX_C) ) * densNew_inv;
#elif NDIM == 3
                Uout_d->at(i, j, k, VELX_VAR_C) = (    Uin_d->at(i,   j,   k,   VELX_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k,   HY_XMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k,   HY_XMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k,   HY_XMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k,   HY_XMOM_FLUX_C)
                                                     + flZ_d->at(i,   j,   k,   HY_XMOM_FLUX_C)
                                                     - flZ_d->at(i,   j,   k+1, HY_XMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELY_VAR_C) = (    Uin_d->at(i,   j,   k,   VELY_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k,   HY_YMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k,   HY_YMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k,   HY_YMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k,   HY_YMOM_FLUX_C)
                                                     + flZ_d->at(i,   j,   k,   HY_YMOM_FLUX_C)
                                                     - flZ_d->at(i,   j,   k+1, HY_YMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, VELZ_VAR_C) = (    Uin_d->at(i,   j,   k,   VELZ_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k,   HY_ZMOM_FLUX_C)
                                                     - flX_d->at(i+1, j,   k,   HY_ZMOM_FLUX_C)
                                                     + flY_d->at(i,   j,   k,   HY_ZMOM_FLUX_C)
                                                     - flY_d->at(i,   j+1, k,   HY_ZMOM_FLUX_C)
                                                     + flZ_d->at(i,   j,   k,   HY_ZMOM_FLUX_C)
                                                     - flZ_d->at(i,   j,   k+1, HY_ZMOM_FLUX_C) ) * densNew_inv;

                Uout_d->at(i, j, k, ENER_VAR_C) = (    Uin_d->at(i,   j,   k,   ENER_VAR_C) * densOld
                                                     + flX_d->at(i,   j,   k,   HY_ENER_FLUX_C)
                                                     - flX_d->at(i+1, j,   k,   HY_ENER_FLUX_C)
                                                     + flY_d->at(i,   j,   k,   HY_ENER_FLUX_C)
                                                     - flY_d->at(i,   j+1, k,   HY_ENER_FLUX_C)
                                                     + flZ_d->at(i,   j,   k,   HY_ENER_FLUX_C)
                                                     - flZ_d->at(i,   j,   k+1, HY_ENER_FLUX_C) ) * densNew_inv;
#endif

#ifdef EINT_VAR_C
                // Compute energy correction from new velocities and energy
                norm2_sqr =   Uout_d->at(i, j, k, VELX_VAR_C) * Uout_d->at(i, j, k, VELX_VAR_C)
                            + Uout_d->at(i, j, k, VELY_VAR_C) * Uout_d->at(i, j, k, VELY_VAR_C)
                            + Uout_d->at(i, j, k, VELZ_VAR_C) * Uout_d->at(i, j, k, VELZ_VAR_C);
                Uout_d->at(i, j, k, EINT_VAR_C) =    Uout_d->at(i, j, k, ENER_VAR_C)
                                                  - (0.5_wp * norm2_sqr);
#endif
            }
        }
    }
}

