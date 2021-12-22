#include "Hydro.h"

#include <Milhoja.h>

#include "Sedov.h"

#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::updateSolutionHll_oacc_summit(const milhoja::IntVect* lo_d,
                                       const milhoja::IntVect* hi_d,
                                       milhoja::FArray4D* U_d,
                                       const milhoja::FArray4D* flX_d,
                                       const milhoja::FArray4D* flY_d,
                                       const milhoja::FArray4D* flZ_d) {
    using namespace milhoja;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

#ifdef EINT_VAR
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
                densOld = U_d->at(i, j, k, DENS_VAR);
#if NDIM == 1
                densNew =   densOld
                          + flX_d->at(i,   j, k, HY_DENS_FLUX)
                          - flX_d->at(i+1, j, k, HY_DENS_FLUX);
#elif NDIM == 2
                densNew =   densOld
                          + flX_d->at(i,   j,   k, HY_DENS_FLUX)
                          - flX_d->at(i+1, j,   k, HY_DENS_FLUX)
                          + flY_d->at(i,   j,   k, HY_DENS_FLUX)
                          - flY_d->at(i,   j+1, k, HY_DENS_FLUX);
#elif NDIM == 3
                densNew =   densOld 
                          + flX_d->at(i,   j,   k,   HY_DENS_FLUX)
                          - flX_d->at(i+1, j,   k,   HY_DENS_FLUX)
                          + flY_d->at(i,   j,   k,   HY_DENS_FLUX)
                          - flY_d->at(i,   j+1, k,   HY_DENS_FLUX)
                          + flZ_d->at(i,   j,   k,   HY_DENS_FLUX)
                          - flZ_d->at(i,   j,   k+1, HY_DENS_FLUX);
#endif
                U_d->at(i, j, k, DENS_VAR) = densNew;
                densNew_inv = 1.0_wp / densNew;

                // velocities and total energy can be updated independently
                // using density result
#if NDIM == 1
                U_d->at(i, j, k, VELX_VAR) = (    U_d->at(i,   j, k, VELX_VAR) * densOld
                                              + flX_d->at(i,   j, k, HY_XMOM_FLUX)
                                              - flX_d->at(i+1, j, k, HY_XMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELY_VAR) = (    U_d->at(i,   j, k, VELY_VAR) * densOld
                                              + flX_d->at(i,   j, k, HY_YMOM_FLUX)
                                              - flX_d->at(i+1, j, k, HY_YMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELZ_VAR) = (    U_d->at(i,   j, k, VELZ_VAR) * densOld
                                              + flX_d->at(i,   j, k, HY_ZMOM_FLUX)
                                              - flX_d->at(i+1, j, k, HY_ZMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, ENER_VAR) = (    U_d->at(i,   j, k, ENER_VAR) * densOld
                                              + flX_d->at(i,   j, k, HY_ENER_FLUX)
                                              - flX_d->at(i+1, j, k, HY_ENER_FLUX) ) * densNew_inv;
#elif NDIM == 2
                U_d->at(i, j, k, VELX_VAR) = (    U_d->at(i,   j,   k, VELX_VAR) * densOld
                                              + flX_d->at(i,   j,   k, HY_XMOM_FLUX)
                                              - flX_d->at(i+1, j,   k, HY_XMOM_FLUX)
                                              + flY_d->at(i,   j,   k, HY_XMOM_FLUX)
                                              - flY_d->at(i,   j+1, k, HY_XMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELY_VAR) = (    U_d->at(i,   j,   k, VELY_VAR) * densOld
                                              + flX_d->at(i,   j,   k, HY_YMOM_FLUX)
                                              - flX_d->at(i+1, j,   k, HY_YMOM_FLUX)
                                              + flY_d->at(i,   j,   k, HY_YMOM_FLUX)
                                              - flY_d->at(i,   j+1, k, HY_YMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELZ_VAR) = (    U_d->at(i,   j,   k, VELZ_VAR) * densOld
                                              + flX_d->at(i,   j,   k, HY_ZMOM_FLUX)
                                              - flX_d->at(i+1, j,   k, HY_ZMOM_FLUX)
                                              + flY_d->at(i,   j,   k, HY_ZMOM_FLUX)
                                              - flY_d->at(i,   j+1, k, HY_ZMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, ENER_VAR) = (    U_d->at(i,   j,   k, ENER_VAR) * densOld
                                              + flX_d->at(i,   j,   k, HY_ENER_FLUX)
                                              - flX_d->at(i+1, j,   k, HY_ENER_FLUX)
                                              + flY_d->at(i,   j,   k, HY_ENER_FLUX)
                                              - flY_d->at(i,   j+1, k, HY_ENER_FLUX) ) * densNew_inv;
#elif NDIM == 3
                U_d->at(i, j, k, VELX_VAR) = (    U_d->at(i,   j,   k,   VELX_VAR) * densOld
                                              + flX_d->at(i,   j,   k,   HY_XMOM_FLUX)
                                              - flX_d->at(i+1, j,   k,   HY_XMOM_FLUX)
                                              + flY_d->at(i,   j,   k,   HY_XMOM_FLUX)
                                              - flY_d->at(i,   j+1, k,   HY_XMOM_FLUX)
                                              + flZ_d->at(i,   j,   k,   HY_XMOM_FLUX)
                                              - flZ_d->at(i,   j,   k+1, HY_XMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELY_VAR) = (    U_d->at(i,   j,   k,   VELY_VAR) * densOld
                                              + flX_d->at(i,   j,   k,   HY_YMOM_FLUX)
                                              - flX_d->at(i+1, j,   k,   HY_YMOM_FLUX)
                                              + flY_d->at(i,   j,   k,   HY_YMOM_FLUX)
                                              - flY_d->at(i,   j+1, k,   HY_YMOM_FLUX)
                                              + flZ_d->at(i,   j,   k,   HY_YMOM_FLUX)
                                              - flZ_d->at(i,   j,   k+1, HY_YMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, VELZ_VAR) = (    U_d->at(i,   j,   k,   VELZ_VAR) * densOld
                                              + flX_d->at(i,   j,   k,   HY_ZMOM_FLUX)
                                              - flX_d->at(i+1, j,   k,   HY_ZMOM_FLUX)
                                              + flY_d->at(i,   j,   k,   HY_ZMOM_FLUX)
                                              - flY_d->at(i,   j+1, k,   HY_ZMOM_FLUX)
                                              + flZ_d->at(i,   j,   k,   HY_ZMOM_FLUX)
                                              - flZ_d->at(i,   j,   k+1, HY_ZMOM_FLUX) ) * densNew_inv;

                U_d->at(i, j, k, ENER_VAR) = (    U_d->at(i,   j,   k,   ENER_VAR) * densOld
                                              + flX_d->at(i,   j,   k,   HY_ENER_FLUX)
                                              - flX_d->at(i+1, j,   k,   HY_ENER_FLUX)
                                              + flY_d->at(i,   j,   k,   HY_ENER_FLUX)
                                              - flY_d->at(i,   j+1, k,   HY_ENER_FLUX)
                                              + flZ_d->at(i,   j,   k,   HY_ENER_FLUX)
                                              - flZ_d->at(i,   j,   k+1, HY_ENER_FLUX) ) * densNew_inv;
#endif

#ifdef EINT_VAR
                // Compute energy correction from new velocities and energy
                norm2_sqr =   U_d->at(i, j, k, VELX_VAR) * U_d->at(i, j, k, VELX_VAR)
                            + U_d->at(i, j, k, VELY_VAR) * U_d->at(i, j, k, VELY_VAR)
                            + U_d->at(i, j, k, VELZ_VAR) * U_d->at(i, j, k, VELZ_VAR);
                U_d->at(i, j, k, EINT_VAR) =    U_d->at(i, j, k, ENER_VAR)
                                             - (0.5_wp * norm2_sqr);
#endif
            }
        }
    }
}

