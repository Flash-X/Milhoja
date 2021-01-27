#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_Z_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flZ_d,
                                        const orchestration::FArray4D* auxC_d) {
    using namespace orchestration;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    Real    sL = 0.0;
    Real    sR = 0.0;
    Real    sRsL = 0.0;
    Real    vn = 0.0;
    Real    vL = 0.0;
    Real    vR = 0.0;
    int     ks = 0;
    int     kL = 0;
    int     kR = 0;
    Real    dtdz = *dt_d / deltas_d->K();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e+K3D; ++k) {
        for     (int j=j_s; j<=j_e;     ++j) {
            for (int i=i_s; i<=i_e;     ++i) {
                sL = std::min(U_d->at(i, j, k-1, VELZ_VAR_C) - auxC_d->at(i, j, k-1, 0),
                              U_d->at(i, j, k,   VELZ_VAR_C) - auxC_d->at(i, j, k,   0));
                sR = std::max(U_d->at(i, j, k-1, VELZ_VAR_C) + auxC_d->at(i, j, k-1, 0),
                              U_d->at(i, j, k,   VELZ_VAR_C) + auxC_d->at(i, j, k,   0));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U_d->at(i, j, k-1, VELZ_VAR_C);
                    ks = k - 1;
                    kL = k - 1;
                    kR = k - 1;
                } else if (sR < 0.0) {
                    vn = U_d->at(i, j, k, VELZ_VAR_C);
                    ks = k;
                    kL = k;
                    kR = k;
                } else {
                    vn = 0.5_wp * (  U_d->at(i, j, k-1, VELZ_VAR_C)
                                   + U_d->at(i, j, k,   VELZ_VAR_C));
                    ks = k;
                    kL = k-1;
                    kR = k;
                    if (vn > 0.0) {
                      --ks;
                    }
                }

                vL = U_d->at(i, j, kL, VELZ_VAR_C);
                vR = U_d->at(i, j, kR, VELZ_VAR_C);
                if (kL == kR) {
                    flZ_d->at(i, j, k, HY_DENS_FLUX_C) =   vn * U_d->at(i, j, ks, DENS_VAR_C);
                    flZ_d->at(i, j, k, HY_XMOM_FLUX_C) =   vn * U_d->at(i, j, ks, DENS_VAR_C)
                                                              * U_d->at(i, j, ks, VELX_VAR_C);
                    flZ_d->at(i, j, k, HY_YMOM_FLUX_C) =   vn * U_d->at(i, j, ks, DENS_VAR_C)
                                                              * U_d->at(i, j, ks, VELY_VAR_C);
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX_C) =   vn * U_d->at(i, j, ks, DENS_VAR_C)
                                                              * U_d->at(i, j, ks, VELZ_VAR_C)
                                                         +      U_d->at(i, j, ks, PRES_VAR_C);
                    flZ_d->at(i, j, k, HY_ENER_FLUX_C) =   vn * U_d->at(i, j, ks, DENS_VAR_C)
                                                              * U_d->at(i, j, ks, ENER_VAR_C)
                                                         + vn * U_d->at(i, j, ks, PRES_VAR_C);
                } else {
                    flZ_d->at(i, j, k, HY_DENS_FLUX_C)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, DENS_VAR_C)
                                                           + sR*sL*(   U_d->at(i, j, kR, DENS_VAR_C)
                                                                    -  U_d->at(i, j, kL, DENS_VAR_C))) /sRsL;
                    flZ_d->at(i, j, k, HY_XMOM_FLUX_C)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELX_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELX_VAR_C)
                                                           + sR*sL*(   U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELX_VAR_C)
                                                                    -  U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELX_VAR_C)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_YMOM_FLUX_C)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELY_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELY_VAR_C)
                                                           + sR*sL*(   U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELY_VAR_C)
                                                                    -  U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELY_VAR_C)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX_C)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELZ_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELZ_VAR_C)
                                                           + sR*sL*(   U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, VELZ_VAR_C)
                                                                    -  U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, VELZ_VAR_C)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX_C) += (  sR * U_d->at(i, j, kL, PRES_VAR_C)
                                                           - sL * U_d->at(i, j, kR, PRES_VAR_C) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ENER_FLUX_C)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, ENER_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, ENER_VAR_C)
                                                           + sR*sL*(   U_d->at(i, j, kR, DENS_VAR_C) * U_d->at(i, j, kR, ENER_VAR_C)
                                                                    -  U_d->at(i, j, kL, DENS_VAR_C) * U_d->at(i, j, kL, ENER_VAR_C))) /sRsL;
                    flZ_d->at(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * U_d->at(i, j, kL, PRES_VAR_C)
                                                           - sL * vR * U_d->at(i, j, kR, PRES_VAR_C) ) /sRsL;
                }

                flZ_d->at(i, j, k, HY_DENS_FLUX_C) *= dtdz;
                flZ_d->at(i, j, k, HY_XMOM_FLUX_C) *= dtdz;
                flZ_d->at(i, j, k, HY_YMOM_FLUX_C) *= dtdz;
                flZ_d->at(i, j, k, HY_ZMOM_FLUX_C) *= dtdz;
                flZ_d->at(i, j, k, HY_ENER_FLUX_C) *= dtdz;
            }
        }
    }
}

