#include "Hydro.h"

#include "Flash.h"

void hy::updateSolutionHll(const orchestration::IntVect& lo,
                           const orchestration::IntVect& hi,
                           orchestration::FArray4D& U,
                           const orchestration::FArray4D& flX,
                           const orchestration::FArray4D& flY,
                           const orchestration::FArray4D& flZ) {
    using namespace orchestration;

#ifdef EINT_VAR_C
    Real    norm2_sqr = 0.0_wp;
#endif
    Real    densOld = 0.0_wp;
    Real    densNew = 0.0_wp;
    Real    densNew_inv = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                // Update density first
                densOld = U(i, j, k, DENS_VAR_C);
#if NDIM == 1
                densNew =   densOld
                          + flX(i,   j, k, HY_DENS_FLUX_C)
                          - flX(i+1, j, k, HY_DENS_FLUX_C);
#elif NDIM == 2
                densNew =   densOld
                          + flX(i,   j,   k, HY_DENS_FLUX_C)
                          - flX(i+1, j,   k, HY_DENS_FLUX_C)
                          + flY(i,   j,   k, HY_DENS_FLUX_C)
                          - flY(i,   j+1, k, HY_DENS_FLUX_C);
#elif NDIM == 3
                densNew =   densOld 
                          + flX(i,   j,   k,   HY_DENS_FLUX_C)
                          - flX(i+1, j,   k,   HY_DENS_FLUX_C)
                          + flY(i,   j,   k,   HY_DENS_FLUX_C)
                          - flY(i,   j+1, k,   HY_DENS_FLUX_C)
                          + flZ(i,   j,   k,   HY_DENS_FLUX_C)
                          - flZ(i,   j,   k+1, HY_DENS_FLUX_C);
#endif
                U(i, j, k, DENS_VAR_C) = densNew;
                densNew_inv = 1.0_wp / densNew;

                // velocities and total energy can be updated independently
                // using density result
#if NDIM == 1
                U(i, j, k, VELX_VAR_C) = (  U(i, j, k, VELX_VAR_C) * densOld
                                          + flX(i,   j, k, HY_XMOM_FLUX_C)
                                          - flX(i+1, j, k, HY_XMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELY_VAR_C) = (  U(i, j, k, VELY_VAR_C) * densOld
                                          + flX(i,   j, k, HY_YMOM_FLUX_C)
                                          - flX(i+1, j, k, HY_YMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELZ_VAR_C) = (  U(i, j, k, VELZ_VAR_C) * densOld
                                          + flX(i,   j, k, HY_ZMOM_FLUX_C)
                                          - flX(i+1, j, k, HY_ZMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, ENER_VAR_C) = (  U(i, j, k, ENER_VAR_C) * densOld
                                          + flX(i,   j, k, HY_ENER_FLUX_C)
                                          - flX(i+1, j, k, HY_ENER_FLUX_C) ) * densNew_inv;
#elif NDIM == 2
                U(i, j, k, VELX_VAR_C) = (  U(i, j, k, VELX_VAR_C) * densOld
                                          + flX(i,   j,   k, HY_XMOM_FLUX_C)
                                          - flX(i+1, j,   k, HY_XMOM_FLUX_C)
                                          + flY(i,   j,   k, HY_XMOM_FLUX_C)
                                          - flY(i,   j+1, k, HY_XMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELY_VAR_C) = (  U(i, j, k, VELY_VAR_C) * densOld
                                          + flX(i,   j,   k, HY_YMOM_FLUX_C)
                                          - flX(i+1, j,   k, HY_YMOM_FLUX_C)
                                          + flY(i,   j,   k, HY_YMOM_FLUX_C)
                                          - flY(i,   j+1, k, HY_YMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELZ_VAR_C) = (  U(i, j, k, VELZ_VAR_C) * densOld
                                          + flX(i,   j,   k, HY_ZMOM_FLUX_C)
                                          - flX(i+1, j,   k, HY_ZMOM_FLUX_C)
                                          + flY(i,   j,   k, HY_ZMOM_FLUX_C)
                                          - flY(i,   j+1, k, HY_ZMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, ENER_VAR_C) = (  U(i, j, k, ENER_VAR_C) * densOld
                                          + flX(i,   j,   k, HY_ENER_FLUX_C)
                                          - flX(i+1, j,   k, HY_ENER_FLUX_C)
                                          + flY(i,   j,   k, HY_ENER_FLUX_C)
                                          - flY(i,   j+1, k, HY_ENER_FLUX_C) ) * densNew_inv;
#elif NDIM == 3
                U(i, j, k, VELX_VAR_C) = (  U(i, j, k, VELX_VAR_C) * densOld
                                          + flX(i,   j,   k,   HY_XMOM_FLUX_C)
                                          - flX(i+1, j,   k,   HY_XMOM_FLUX_C)
                                          + flY(i,   j,   k,   HY_XMOM_FLUX_C)
                                          - flY(i,   j+1, k,   HY_XMOM_FLUX_C)
                                          + flZ(i,   j,   k,   HY_XMOM_FLUX_C)
                                          - flZ(i,   j,   k+1, HY_XMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELY_VAR_C) = (  U(i, j, k, VELY_VAR_C) * densOld
                                          + flX(i,   j,   k,   HY_YMOM_FLUX_C)
                                          - flX(i+1, j,   k,   HY_YMOM_FLUX_C)
                                          + flY(i,   j,   k,   HY_YMOM_FLUX_C)
                                          - flY(i,   j+1, k,   HY_YMOM_FLUX_C)
                                          + flZ(i,   j,   k,   HY_YMOM_FLUX_C)
                                          - flZ(i,   j,   k+1, HY_YMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, VELZ_VAR_C) = (  U(i, j, k, VELZ_VAR_C) * densOld
                                          + flX(i,   j,   k,   HY_ZMOM_FLUX_C)
                                          - flX(i+1, j,   k,   HY_ZMOM_FLUX_C)
                                          + flY(i,   j,   k,   HY_ZMOM_FLUX_C)
                                          - flY(i,   j+1, k,   HY_ZMOM_FLUX_C)
                                          + flZ(i,   j,   k,   HY_ZMOM_FLUX_C)
                                          - flZ(i,   j,   k+1, HY_ZMOM_FLUX_C) ) * densNew_inv;

                U(i, j, k, ENER_VAR_C) = (  U(i, j, k, ENER_VAR_C) * densOld
                                          + flX(i,   j,   k,   HY_ENER_FLUX_C)
                                          - flX(i+1, j,   k,   HY_ENER_FLUX_C)
                                          + flY(i,   j,   k,   HY_ENER_FLUX_C)
                                          - flY(i,   j+1, k,   HY_ENER_FLUX_C)
                                          + flZ(i,   j,   k,   HY_ENER_FLUX_C)
                                          - flZ(i,   j,   k+1, HY_ENER_FLUX_C) ) * densNew_inv;
#endif

#ifdef EINT_VAR_C
                // Compute energy correction from new velocities and energy
                norm2_sqr =   U(i, j, k, VELX_VAR_C) * U(i, j, k, VELX_VAR_C)
                            + U(i, j, k, VELY_VAR_C) * U(i, j, k, VELY_VAR_C)
                            + U(i, j, k, VELZ_VAR_C) * U(i, j, k, VELZ_VAR_C);
                U(i, j, k, EINT_VAR_C) =    U(i, j, k, ENER_VAR_C)
                                         - (0.5_wp * norm2_sqr);
#endif
            }
        }
    }
}

