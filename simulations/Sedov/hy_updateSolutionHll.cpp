#include "Hydro.h"

#include "Flash.h"

void hy::updateSolutionHll(const orchestration::IntVect& lo,
                           const orchestration::IntVect& hi,
                           const orchestration::FArray4D& Uin,
                           orchestration::FArray4D& Uout,
                           const orchestration::FArray4D& flX,
                           const orchestration::FArray4D& flY,
                           const orchestration::FArray4D& flZ) {
    using namespace orchestration;

//  if (hy_fluxCorrect) then
//     call Driver_abortFlash("hy_hllUnsplit: flux correction is not implemented!")
//  end if
//
//  if (hy_useGravity) then
//     call Driver_abortFlash("hy_hllUnsplit: support for gravity not implemented!")
//  end if
//
//  if (.NOT.hy_updateHydroFluxes) then
//     return
//  end if

    //************************************************************************
    // Unsplit update for conservative variables from n to n+1 time step
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                Uout(i, j, k, VELX_VAR_C) =   Uin(i,   j, k, VELX_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, VELY_VAR_C) =   Uin(i,   j, k, VELY_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, VELZ_VAR_C) =   Uin(i,   j, k, VELZ_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, ENER_VAR_C) =   Uin(i,   j, k, ENER_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, DENS_VAR_C) =   Uin(i,   j, k, DENS_VAR_C)
                                            + flX(i,   j, k, HY_DENS_FLUX_C)
                                            - flX(i+1, j, k, HY_DENS_FLUX_C);
                // TODO: After we get timing data on this code, replace the branches
                // here with preprocessor macros to get an idea of the performance
                // hit of the branch statements.
                if (NDIM > 1) {
                     Uout(i, j, k, DENS_VAR_C) += (  flY(i, j,   k, HY_DENS_FLUX_C)
                                                   - flY(i, j+1, k, HY_DENS_FLUX_C) );
                }
                if (NDIM > 2) {
                     Uout(i, j, k, DENS_VAR_C) += (  flZ(i, j, k,   HY_DENS_FLUX_C)
                                                   - flZ(i, j, k+1, HY_DENS_FLUX_C) );
                }
            }
        }
    }
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                // TODO: After we get timing data on this code, fuse this code
                //       with previous loop nest.
                Uout(i, j, k, VELX_VAR_C) += (  flX(i,   j, k, HY_XMOM_FLUX_C)
                                              - flX(i+1, j, k, HY_XMOM_FLUX_C) );
                Uout(i, j, k, VELY_VAR_C) += (  flX(i,   j, k, HY_YMOM_FLUX_C)
                                              - flX(i+1, j, k, HY_YMOM_FLUX_C) );
                Uout(i, j, k, VELZ_VAR_C) += (  flX(i,   j, k, HY_ZMOM_FLUX_C)
                                              - flX(i+1, j, k, HY_ZMOM_FLUX_C) );
                Uout(i, j, k, ENER_VAR_C) += (  flX(i,   j, k, HY_ENER_FLUX_C)
                                              - flX(i+1, j, k, HY_ENER_FLUX_C) );
            }
        }
    }
#if NDIM > 1
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                Uout(i, j, k, VELX_VAR_C) += (  flY(i, j,   k, HY_XMOM_FLUX_C)
                                              - flY(i, j+1, k, HY_XMOM_FLUX_C) );
                Uout(i, j, k, VELY_VAR_C) += (  flY(i, j,   k, HY_YMOM_FLUX_C)
                                              - flY(i, j+1, k, HY_YMOM_FLUX_C) );
                Uout(i, j, k, VELZ_VAR_C) += (  flY(i, j,   k, HY_ZMOM_FLUX_C)
                                              - flY(i, j+1, k, HY_ZMOM_FLUX_C) );
                Uout(i, j, k, ENER_VAR_C) += (  flY(i, j,   k, HY_ENER_FLUX_C)
                                              - flY(i, j+1, k, HY_ENER_FLUX_C) );
            }
        }
    }
#endif
#if NDIM > 2
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
           Uout(i, j, k, VELX_VAR_C) += (  flZ(i, j, k,   HY_XMOM_FLUX_C)
                                         - flZ(i, j, k+1, HY_XMOM_FLUX_C) );
           Uout(i, j, k, VELY_VAR_C) += (  flZ(i, j, k,   HY_YMOM_FLUX_C)
                                         - flZ(i, j, k+1, HY_YMOM_FLUX_C) );
           Uout(i, j, k, VELZ_VAR_C) += (  flZ(i, j, k,   HY_ZMOM_FLUX_C)
                                         - flZ(i, j, k+1, HY_ZMOM_FLUX_C) );
           Uout(i, j, k, ENER_VAR_C) += (  flZ(i, j, k,   HY_ENER_FLUX_C)
                                         - flZ(i, j, k+1, HY_ENER_FLUX_C) );
            }
        }
    }
#endif

    Real   norm2_sqr = 0.0_wp;
    Real   invNewDens = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                invNewDens = 1.0_wp / Uout(i, j, k, DENS_VAR_C);
                Uout(i, j, k, VELX_VAR_C) *= invNewDens;
                Uout(i, j, k, VELY_VAR_C) *= invNewDens;
                Uout(i, j, k, VELZ_VAR_C) *= invNewDens;
                Uout(i, j, k, ENER_VAR_C) *= invNewDens;
#ifdef EINT_VAR_C
                // Correct energy if necessary
                norm2_sqr =   Uout(i, j, k, VELX_VAR_C) * Uout(i, j, k, VELX_VAR_C)
                            + Uout(i, j, k, VELY_VAR_C) * Uout(i, j, k, VELY_VAR_C)
                            + Uout(i, j, k, VELZ_VAR_C) * Uout(i, j, k, VELZ_VAR_C);
                Uout(i, j, k, EINT_VAR_C) =    Uout(i, j, k, ENER_VAR_C)
                                            - (0.5_wp * norm2_sqr);
#endif
            }
        }
    }
}

