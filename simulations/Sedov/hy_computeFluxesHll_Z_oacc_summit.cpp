#include "Hydro.h"

#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_Z_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flZ,
                                        const orchestration::FArray3D& auxC) {
    using namespace orchestration;

    Real    sL = 0.0;
    Real    sR = 0.0;
    Real    sRsL = 0.0;
    Real    vn = 0.0;
    Real    vL = 0.0;
    Real    vR = 0.0;
    int     ks = 0;
    int     kL = 0;
    int     kR = 0;
    Real    dtdz = dt / deltas.K();

    for         (int k=lo.K(); k<=hi.K()+K3D; ++k) {
        for     (int j=lo.J(); j<=hi.J();     ++j) {
            for (int i=lo.I(); i<=hi.I();     ++i) {
                sL = std::min(U(i, j, k-1, VELZ_VAR_C) - auxC(i, j, k-1),
                              U(i, j, k,   VELZ_VAR_C) - auxC(i, j, k));
                sR = std::max(U(i, j, k-1, VELZ_VAR_C) + auxC(i, j, k-1),
                              U(i, j, k,   VELZ_VAR_C) + auxC(i, j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U(i, j, k-1, VELZ_VAR_C);
                    ks = k - 1;
                    kL = k - 1;
                    kR = k - 1;
                } else if (sR < 0.0) {
                    vn = U(i, j, k, VELZ_VAR_C);
                    ks = k;
                    kL = k;
                    kR = k;
                } else {
                    vn = 0.5_wp * (  U(i, j, k-1, VELZ_VAR_C)
                                   + U(i, j, k,   VELZ_VAR_C));
                    ks = k;
                    kL = k-1;
                    kR = k;
                    if (vn > 0.0) {
                      --ks;
                    }
                }

                vL = U(i, j, kL, VELZ_VAR_C);
                vR = U(i, j, kR, VELZ_VAR_C);
                if (kL == kR) {
                    flZ(i, j, k, HY_DENS_FLUX_C) =   vn * U(i, j, ks, DENS_VAR_C);
                    flZ(i, j, k, HY_XMOM_FLUX_C) =   vn * U(i, j, ks, DENS_VAR_C)
                                                        * U(i, j, ks, VELX_VAR_C);
                    flZ(i, j, k, HY_YMOM_FLUX_C) =   vn * U(i, j, ks, DENS_VAR_C)
                                                        * U(i, j, ks, VELY_VAR_C);
                    flZ(i, j, k, HY_ZMOM_FLUX_C) =   vn * U(i, j, ks, DENS_VAR_C)
                                                        * U(i, j, ks, VELZ_VAR_C)
                                                   +      U(i, j, ks, PRES_VAR_C);
                    flZ(i, j, k, HY_ENER_FLUX_C) =   vn * U(i, j, ks, DENS_VAR_C)
                                                        * U(i, j, ks, ENER_VAR_C)
                                                   + vn * U(i, j, ks, PRES_VAR_C);
                } else {
                    flZ(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * U(i, j, kL, DENS_VAR_C)
                                                    - sL * vR * U(i, j, kR, DENS_VAR_C)
                                                    + sR*sL*(   U(i, j, kR, DENS_VAR_C)
                                                             -  U(i, j, kL, DENS_VAR_C))) /sRsL;
                    flZ(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELX_VAR_C)
                                                    - sL * vR * U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELX_VAR_C)
                                                    + sR*sL*(   U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELX_VAR_C)
                                                             -  U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELX_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELY_VAR_C)
                                                    - sL * vR * U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELY_VAR_C)
                                                    + sR*sL*(   U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELY_VAR_C)
                                                             -  U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELY_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELZ_VAR_C)
                                                    - sL * vR * U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELZ_VAR_C)
                                                    + sR*sL*(   U(i, j, kR, DENS_VAR_C) * U(i, j, kR, VELZ_VAR_C)
                                                             -  U(i, j, kL, DENS_VAR_C) * U(i, j, kL, VELZ_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX_C) += (  sR * U(i, j, kL, PRES_VAR_C)
                                                     - sL * U(i, j, kR, PRES_VAR_C) ) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * U(i, j, kL, DENS_VAR_C) * U(i, j, kL, ENER_VAR_C)
                                                    - sL * vR * U(i, j, kR, DENS_VAR_C) * U(i, j, kR, ENER_VAR_C)
                                                    + sR*sL*(   U(i, j, kR, DENS_VAR_C) * U(i, j, kR, ENER_VAR_C)
                                                             -  U(i, j, kL, DENS_VAR_C) * U(i, j, kL, ENER_VAR_C))) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * U(i, j, kL, PRES_VAR_C)
                                                     - sL * vR * U(i, j, kR, PRES_VAR_C) ) /sRsL;
                }

                flZ(i, j, k, HY_DENS_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_XMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_YMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_ZMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_ENER_FLUX_C) *= dtdz;
            }
        }
    }
}

