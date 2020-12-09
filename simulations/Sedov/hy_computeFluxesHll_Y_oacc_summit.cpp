#include "Hydro.h"

#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_Y_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flY,
                                        const orchestration::FArray3D& auxC) {
    using namespace orchestration;

    Real    sL = 0.0;
    Real    sR = 0.0;
    Real    sRsL = 0.0;
    Real    vn = 0.0;
    Real    vL = 0.0;
    Real    vR = 0.0;
    int     js = 0;
    int     jL = 0;
    int     jR = 0;
    Real    dtdy = dt / deltas.J();

    for         (int k=lo.K(); k<=hi.K();     ++k) {
        for     (int j=lo.J(); j<=hi.J()+K2D; ++j) {
            for (int i=lo.I(); i<=hi.I();     ++i) {
                sL = std::min(U(i, j-1, k, VELY_VAR_C) - auxC(i, j-1, k),
                              U(i, j,   k, VELY_VAR_C) - auxC(i, j,   k));
                sR = std::max(U(i, j-1, k, VELY_VAR_C) + auxC(i, j-1, k),
                              U(i, j,   k, VELY_VAR_C) + auxC(i, j,   k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U(i, j-1, k, VELY_VAR_C);
                    js = j - 1;
                    jL = j - 1;
                    jR = j - 1;
                } else if (sR < 0.0) {
                    vn = U(i, j, k, VELY_VAR_C);
                    js = j;
                    jL = j;
                    jR = j;
                } else {
                    vn = 0.5_wp * (U(i, j-1, k, VELY_VAR_C) + U(i, j, k, VELY_VAR_C));
                    js = j;
                    jL = j - 1;
                    jR = j;
                    if (vn > 0.0) {
                        --js;
                    }
                }

                vL = U(i, jL, k, VELY_VAR_C);
                vR = U(i, jR, k, VELY_VAR_C);
                if (jL == jR) {
                    flY(i, j, k, HY_DENS_FLUX_C) =   vn * U(i, js, k, DENS_VAR_C);
                    flY(i, j, k, HY_XMOM_FLUX_C) =   vn * U(i, js, k, DENS_VAR_C)
                                                        * U(i, js, k, VELX_VAR_C);
                    flY(i, j, k, HY_YMOM_FLUX_C) =   vn * U(i, js, k, DENS_VAR_C)
                                                        * U(i, js, k, VELY_VAR_C)
                                                   +      U(i, js, k, PRES_VAR_C);
                    flY(i, j, k, HY_ZMOM_FLUX_C) =   vn * U(i, js, k, DENS_VAR_C)
                                                        * U(i, js, k, VELZ_VAR_C);
                    flY(i, j, k, HY_ENER_FLUX_C) =   vn * U(i, js, k, DENS_VAR_C)
                                                        * U(i, js, k, ENER_VAR_C)
                                                   + vn * U(i,js,k,PRES_VAR_C);
                } else {
                    flY(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * U(i, jL, k, DENS_VAR_C)
                                                    - sL * vR * U(i, jR, k, DENS_VAR_C)
                                                    + sR*sL*(   U(i, jR, k, DENS_VAR_C)
                                                             -  U(i, jL, k, DENS_VAR_C))) /sRsL;
                    flY(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELX_VAR_C)
                                                    - sL * vR * U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELX_VAR_C)
                                                    + sR*sL*(   U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELX_VAR_C)
                                                             -  U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELX_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELY_VAR_C)
                                                    - sL * vR * U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELY_VAR_C)
                                                    + sR*sL*(   U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELY_VAR_C)
                                                             -  U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELY_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX_C) +=  (  sR * U(i, jL, k, PRES_VAR_C)
                                                      - sL * U(i, jR, k, PRES_VAR_C) ) / sRsL;
                    flY(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELZ_VAR_C)
                                                    - sL * vR * U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELZ_VAR_C)
                                                    + sR*sL*(   U(i, jR, k, DENS_VAR_C) * U(i, jR, k, VELZ_VAR_C)
                                                             -  U(i, jL, k, DENS_VAR_C) * U(i, jL, k, VELZ_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * U(i, jL, k, DENS_VAR_C) * U(i, jL, k, ENER_VAR_C)
                                                    - sL * vR * U(i, jR, k, DENS_VAR_C) * U(i, jR, k, ENER_VAR_C)
                                                    + sR*sL*(   U(i, jR, k, DENS_VAR_C) * U(i, jR, k, ENER_VAR_C)
                                                             -  U(i, jL, k, DENS_VAR_C) * U(i, jL, k, ENER_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX_C) +=  (  sR * vL * U(i, jL, k, PRES_VAR_C)
                                                      - sL * vR * U(i, jR, k, PRES_VAR_C) ) /sRsL;
                }

                flY(i, j, k, HY_DENS_FLUX_C) *= dtdy; 
                flY(i, j, k, HY_XMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_YMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_ZMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_ENER_FLUX_C) *= dtdy;
            }
        }
    }
}

