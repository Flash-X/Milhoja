#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include <algorithm>

#include "milhoja.h"

#include "Sedov.h"

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
                sL = std::min(U_d->at(i, j, k-1, VELZ_VAR) - auxC_d->at(i, j, k-1, 0),
                              U_d->at(i, j, k,   VELZ_VAR) - auxC_d->at(i, j, k,   0));
                sR = std::max(U_d->at(i, j, k-1, VELZ_VAR) + auxC_d->at(i, j, k-1, 0),
                              U_d->at(i, j, k,   VELZ_VAR) + auxC_d->at(i, j, k,   0));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U_d->at(i, j, k-1, VELZ_VAR);
                    ks = k - 1;
                    kL = k - 1;
                    kR = k - 1;
                } else if (sR < 0.0) {
                    vn = U_d->at(i, j, k, VELZ_VAR);
                    ks = k;
                    kL = k;
                    kR = k;
                } else {
                    vn = 0.5_wp * (  U_d->at(i, j, k-1, VELZ_VAR)
                                   + U_d->at(i, j, k,   VELZ_VAR));
                    ks = k;
                    kL = k-1;
                    kR = k;
                    if (vn > 0.0) {
                      --ks;
                    }
                }

                vL = U_d->at(i, j, kL, VELZ_VAR);
                vR = U_d->at(i, j, kR, VELZ_VAR);
                if (kL == kR) {
                    flZ_d->at(i, j, k, HY_DENS_FLUX) =   vn * U_d->at(i, j, ks, DENS_VAR);
                    flZ_d->at(i, j, k, HY_XMOM_FLUX) =   vn * U_d->at(i, j, ks, DENS_VAR)
                                                            * U_d->at(i, j, ks, VELX_VAR);
                    flZ_d->at(i, j, k, HY_YMOM_FLUX) =   vn * U_d->at(i, j, ks, DENS_VAR)
                                                            * U_d->at(i, j, ks, VELY_VAR);
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX) =   vn * U_d->at(i, j, ks, DENS_VAR)
                                                            * U_d->at(i, j, ks, VELZ_VAR)
                                                       +      U_d->at(i, j, ks, PRES_VAR);
                    flZ_d->at(i, j, k, HY_ENER_FLUX) =   vn * U_d->at(i, j, ks, DENS_VAR)
                                                            * U_d->at(i, j, ks, ENER_VAR)
                                                       + vn * U_d->at(i, j, ks, PRES_VAR);
                } else {
                    flZ_d->at(i, j, k, HY_DENS_FLUX)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, DENS_VAR)
                                                         + sR*sL*(   U_d->at(i, j, kR, DENS_VAR)
                                                                  -  U_d->at(i, j, kL, DENS_VAR))) /sRsL;
                    flZ_d->at(i, j, k, HY_XMOM_FLUX)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELX_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELX_VAR)
                                                         + sR*sL*(   U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELX_VAR)
                                                                  -  U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELX_VAR)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_YMOM_FLUX)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELY_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELY_VAR)
                                                         + sR*sL*(   U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELY_VAR)
                                                                  -  U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELY_VAR)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELZ_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELZ_VAR)
                                                         + sR*sL*(   U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, VELZ_VAR)
                                                                  -  U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, VELZ_VAR)) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ZMOM_FLUX) += (  sR * U_d->at(i, j, kL, PRES_VAR)
                                                         - sL * U_d->at(i, j, kR, PRES_VAR) ) /sRsL;
                    flZ_d->at(i, j, k, HY_ENER_FLUX)  = (  sR * vL * U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, ENER_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, ENER_VAR)
                                                         + sR*sL*(   U_d->at(i, j, kR, DENS_VAR) * U_d->at(i, j, kR, ENER_VAR)
                                                                  -  U_d->at(i, j, kL, DENS_VAR) * U_d->at(i, j, kL, ENER_VAR))) /sRsL;
                    flZ_d->at(i, j, k, HY_ENER_FLUX) += (  sR * vL * U_d->at(i, j, kL, PRES_VAR)
                                                         - sL * vR * U_d->at(i, j, kR, PRES_VAR) ) /sRsL;
                }

                flZ_d->at(i, j, k, HY_DENS_FLUX) *= dtdz;
                flZ_d->at(i, j, k, HY_XMOM_FLUX) *= dtdz;
                flZ_d->at(i, j, k, HY_YMOM_FLUX) *= dtdz;
                flZ_d->at(i, j, k, HY_ZMOM_FLUX) *= dtdz;
                flZ_d->at(i, j, k, HY_ENER_FLUX) *= dtdz;
            }
        }
    }
}

