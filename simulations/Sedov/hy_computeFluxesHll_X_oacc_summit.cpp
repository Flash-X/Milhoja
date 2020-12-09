#include "Hydro.h"

#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_X_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flX,
                                        const orchestration::FArray3D& auxC) {
    using namespace orchestration;

    Real    sL = 0.0;
    Real    sR = 0.0;
    Real    sRsL = 0.0;
    Real    vn = 0.0;
    Real    vL = 0.0;
    Real    vR = 0.0;
    int     is = 0;
    int     iL = 0;
    int     iR = 0;
    Real    dtdx = dt / deltas.I();

    for         (int k=lo.K(); k<=hi.K();     ++k) {
        for     (int j=lo.J(); j<=hi.J();     ++j) {
            for (int i=lo.I(); i<=hi.I()+K1D; ++i) {
                sL = std::min(U(i-1, j, k, VELX_VAR_C) - auxC(i-1, j, k),
                              U(i,   j, k, VELX_VAR_C) - auxC(i,   j, k));
                sR = std::max(U(i-1, j, k, VELX_VAR_C) + auxC(i-1, j, k),
                              U(i,   j, k, VELX_VAR_C) + auxC(i,   j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U(i-1, j, k, VELX_VAR_C);
                    is = i - 1;
                    iL = i - 1;
                    iR = i - 1;
                } else if (sR < 0.0) {
                    vn = U(i, j, k, VELX_VAR_C);
                    is = i;
                    iL = i;
                    iR = i;
                } else {
                    vn = 0.5_wp * (  U(i-1, j, k, VELX_VAR_C)
                                   + U(i,   j, k, VELX_VAR_C));
                    is = i;
                    iL = i-1;
                    iR = i;
                    if (vn > 0.0) {
                        --is;
                    }
                }

                vL = U(iL, j, k, VELX_VAR_C);
                vR = U(iR, j, k, VELX_VAR_C);
                if (iL == iR) {
                    flX(i, j, k, HY_DENS_FLUX_C) =   vn * U(is, j, k, DENS_VAR_C);
                    flX(i, j, k, HY_XMOM_FLUX_C) =   vn * U(is, j, k, DENS_VAR_C)
                                                        * U(is, j, k, VELX_VAR_C)
                                                   +      U(is, j, k, PRES_VAR_C);
                    flX(i, j, k, HY_YMOM_FLUX_C) =   vn * U(is, j, k, DENS_VAR_C)
                                                        * U(is, j, k, VELY_VAR_C);
                    flX(i, j, k, HY_ZMOM_FLUX_C) =   vn * U(is, j, k, DENS_VAR_C)
                                                        * U(is, j, k, VELZ_VAR_C);
                    flX(i, j, k, HY_ENER_FLUX_C) =   vn * U(is, j, k, DENS_VAR_C)
                                                        * U(is, j, k, ENER_VAR_C)
                                                   + vn * U(is, j, k, PRES_VAR_C);
                } else {
                    flX(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * U(iL, j, k, DENS_VAR_C)
                                                    - sL * vR * U(iR, j, k, DENS_VAR_C)
                                                    + sR*sL*(   U(iR, j, k, DENS_VAR_C)
                                                              - U(iL, j, k, DENS_VAR_C)) ) / sRsL;
                    flX(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELX_VAR_C)
                                                    - sL * vR * U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELX_VAR_C)
                                                    + sR*sL*(   U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELX_VAR_C)
                                                              - U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELX_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_XMOM_FLUX_C) += (  sR * U(iL, j, k, PRES_VAR_C)
                                                     - sL * U(iR, j, k, PRES_VAR_C) ) /sRsL;
                    flX(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELY_VAR_C)
                                                    - sL * vR * U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELY_VAR_C)
                                                    + sR*sL*(   U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELY_VAR_C)
                                                              - U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELY_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELZ_VAR_C)
                                                    - sL * vR * U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELZ_VAR_C)
                                                    + sR*sL*(   U(iR, j, k, DENS_VAR_C) * U(iR, j, k, VELZ_VAR_C)
                                                              - U(iL, j, k, DENS_VAR_C) * U(iL, j, k, VELZ_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * U(iL, j, k, DENS_VAR_C) * U(iL, j, k, ENER_VAR_C)
                                                    - sL * vR * U(iR, j, k, DENS_VAR_C) * U(iR, j, k, ENER_VAR_C)
                                                    + sR*sL*(   U(iR, j, k, DENS_VAR_C) * U(iR, j, k, ENER_VAR_C)
                                                              - U(iL, j, k, DENS_VAR_C) * U(iL, j, k, ENER_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * U(iL, j, k, PRES_VAR_C)
                                                     - sL * vR * U(iR, j, k, PRES_VAR_C)) / sRsL;
                }

                flX(i, j, k, HY_DENS_FLUX_C) *= dtdx;
                flX(i, j, k, HY_XMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_YMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_ZMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_ENER_FLUX_C) *= dtdx;
            }
        }
    }
}

