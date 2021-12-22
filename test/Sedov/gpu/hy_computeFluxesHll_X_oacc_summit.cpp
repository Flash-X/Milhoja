#include "Hydro.h"

#include <algorithm>

#include <Milhoja.h>

#include "Sedov.h"

#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::computeFluxesHll_X_oacc_summit(const milhoja::Real* dt_d,
                                        const milhoja::IntVect* lo_d,
                                        const milhoja::IntVect* hi_d,
                                        const milhoja::RealVect* deltas_d,
                                        const milhoja::FArray4D* U_d,
                                        milhoja::FArray4D* flX_d,
                                        const milhoja::FArray4D* auxC_d) {
    using namespace milhoja;

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
    int     is = 0;
    int     iL = 0;
    int     iR = 0;
    Real    dtdx = *dt_d / deltas_d->I();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e;     ++k) {
        for     (int j=j_s; j<=j_e;     ++j) {
            for (int i=i_s; i<=i_e+K1D; ++i) {
                sL = std::min(U_d->at(i-1, j, k, VELX_VAR) - auxC_d->at(i-1, j, k, 0),
                              U_d->at(i,   j, k, VELX_VAR) - auxC_d->at(i,   j, k, 0));
                sR = std::max(U_d->at(i-1, j, k, VELX_VAR) + auxC_d->at(i-1, j, k, 0),
                              U_d->at(i,   j, k, VELX_VAR) + auxC_d->at(i,   j, k, 0));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U_d->at(i-1, j, k, VELX_VAR);
                    is = i - 1;
                    iL = i - 1;
                    iR = i - 1;
                } else if (sR < 0.0) {
                    vn = U_d->at(i, j, k, VELX_VAR);
                    is = i;
                    iL = i;
                    iR = i;
                } else {
                    vn = 0.5_wp * (  U_d->at(i-1, j, k, VELX_VAR)
                                   + U_d->at(i,   j, k, VELX_VAR));
                    is = i;
                    iL = i-1;
                    iR = i;
                    if (vn > 0.0) {
                        --is;
                    }
                }

                vL = U_d->at(iL, j, k, VELX_VAR);
                vR = U_d->at(iR, j, k, VELX_VAR);
                if (iL == iR) {
                    flX_d->at(i, j, k, HY_DENS_FLUX) =   vn * U_d->at(is, j, k, DENS_VAR);
                    flX_d->at(i, j, k, HY_XMOM_FLUX) =   vn * U_d->at(is, j, k, DENS_VAR)
                                                            * U_d->at(is, j, k, VELX_VAR)
                                                       +      U_d->at(is, j, k, PRES_VAR);
                    flX_d->at(i, j, k, HY_YMOM_FLUX) =   vn * U_d->at(is, j, k, DENS_VAR)
                                                            * U_d->at(is, j, k, VELY_VAR);
                    flX_d->at(i, j, k, HY_ZMOM_FLUX) =   vn * U_d->at(is, j, k, DENS_VAR)
                                                            * U_d->at(is, j, k, VELZ_VAR);
                    flX_d->at(i, j, k, HY_ENER_FLUX) =   vn * U_d->at(is, j, k, DENS_VAR)
                                                            * U_d->at(is, j, k, ENER_VAR)
                                                       + vn * U_d->at(is, j, k, PRES_VAR);
                } else {
                    flX_d->at(i, j, k, HY_DENS_FLUX)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, DENS_VAR)
                                                         + sR*sL*(   U_d->at(iR, j, k, DENS_VAR)
                                                                   - U_d->at(iL, j, k, DENS_VAR)) ) / sRsL;
                    flX_d->at(i, j, k, HY_XMOM_FLUX)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELX_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELX_VAR)
                                                         + sR*sL*(   U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELX_VAR)
                                                                   - U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELX_VAR)) )/sRsL;
                    flX_d->at(i, j, k, HY_XMOM_FLUX) += (  sR * U_d->at(iL, j, k, PRES_VAR)
                                                         - sL * U_d->at(iR, j, k, PRES_VAR) ) /sRsL;
                    flX_d->at(i, j, k, HY_YMOM_FLUX)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELY_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELY_VAR)
                                                         + sR*sL*(   U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELY_VAR)
                                                                   - U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELY_VAR)) )/sRsL;
                    flX_d->at(i, j, k, HY_ZMOM_FLUX)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELZ_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELZ_VAR)
                                                         + sR*sL*(   U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, VELZ_VAR)
                                                                   - U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, VELZ_VAR)) )/sRsL;
                    flX_d->at(i, j, k, HY_ENER_FLUX)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, ENER_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, ENER_VAR)
                                                         + sR*sL*(   U_d->at(iR, j, k, DENS_VAR) * U_d->at(iR, j, k, ENER_VAR)
                                                                   - U_d->at(iL, j, k, DENS_VAR) * U_d->at(iL, j, k, ENER_VAR)) )/sRsL;
                    flX_d->at(i, j, k, HY_ENER_FLUX) += (  sR * vL * U_d->at(iL, j, k, PRES_VAR)
                                                         - sL * vR * U_d->at(iR, j, k, PRES_VAR)) / sRsL;
                }

                flX_d->at(i, j, k, HY_DENS_FLUX) *= dtdx;
                flX_d->at(i, j, k, HY_XMOM_FLUX) *= dtdx;
                flX_d->at(i, j, k, HY_YMOM_FLUX) *= dtdx;
                flX_d->at(i, j, k, HY_ZMOM_FLUX) *= dtdx;
                flX_d->at(i, j, k, HY_ENER_FLUX) *= dtdx;
            }
        }
    }
}

