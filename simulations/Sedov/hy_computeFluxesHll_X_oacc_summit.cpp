#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

//#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_X_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flX_d,
                                        const orchestration::FArray4D* auxC_d) {
    using namespace orchestration;

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

//    Real    sL = 0.0;
//    Real    sR = 0.0;
//    Real    sRsL = 0.0;
//    Real    vn = 0.0;
//    Real    vL = 0.0;
//    Real    vR = 0.0;
//    int     is = 0;
//    int     iL = 0;
//    int     iR = 0;
    Real    dtdx = *dt_d / deltas_d->I();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e;     ++k) {
        for     (int j=j_s; j<=j_e;     ++j) {
            for (int i=i_s; i<=i_e+K1D; ++i) {
                flX_d->at(i, j, k, HY_XMOM_FLUX_C) = dtdx;
//                sL = std::min(U_d->at(i-1, j, k, VELX_VAR_C) - auxC_d->at(i-1, j, k, 0),
//                              U_d->at(i,   j, k, VELX_VAR_C) - auxC_d->at(i,   j, k, 0));
//                sR = std::max(U_d->at(i-1, j, k, VELX_VAR_C) + auxC_d->at(i-1, j, k, 0),
//                              U_d->at(i,   j, k, VELX_VAR_C) + auxC_d->at(i,   j, k, 0));
//                sRsL = sR - sL;
//                if (sL > 0.0) {
//                    vn = U_d->at(i-1, j, k, VELX_VAR_C);
//                    is = i - 1;
//                    iL = i - 1;
//                    iR = i - 1;
//                } else if (sR < 0.0) {
//                    vn = U_d->at(i, j, k, VELX_VAR_C);
//                    is = i;
//                    iL = i;
//                    iR = i;
//                } else {
//                    vn = 0.5_wp * (  U_d->at(i-1, j, k, VELX_VAR_C)
//                                   + U_d->at(i,   j, k, VELX_VAR_C));
//                    is = i;
//                    iL = i-1;
//                    iR = i;
//                    if (vn > 0.0) {
//                        --is;
//                    }
//                }
//
//                vL = U_d->at(iL, j, k, VELX_VAR_C);
//                vR = U_d->at(iR, j, k, VELX_VAR_C);
//                if (iL == iR) {
//                    flX_d->at(i, j, k, HY_DENS_FLUX_C) =   vn * U_d->at(is, j, k, DENS_VAR_C);
//                    flX_d->at(i, j, k, HY_XMOM_FLUX_C) =   vn * U_d->at(is, j, k, DENS_VAR_C)
//                                                              * U_d->at(is, j, k, VELX_VAR_C)
//                                                         +      U_d->at(is, j, k, PRES_VAR_C);
//                    flX_d->at(i, j, k, HY_YMOM_FLUX_C) =   vn * U_d->at(is, j, k, DENS_VAR_C)
//                                                              * U_d->at(is, j, k, VELY_VAR_C);
//                    flX_d->at(i, j, k, HY_ZMOM_FLUX_C) =   vn * U_d->at(is, j, k, DENS_VAR_C)
//                                                              * U_d->at(is, j, k, VELZ_VAR_C);
//                    flX_d->at(i, j, k, HY_ENER_FLUX_C) =   vn * U_d->at(is, j, k, DENS_VAR_C)
//                                                              * U_d->at(is, j, k, ENER_VAR_C)
//                                                         + vn * U_d->at(is, j, k, PRES_VAR_C);
//                } else {
//                    flX_d->at(i, j, k, HY_DENS_FLUX_C)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, DENS_VAR_C)
//                                                           + sR*sL*(   U_d->at(iR, j, k, DENS_VAR_C)
//                                                                     - U_d->at(iL, j, k, DENS_VAR_C)) ) / sRsL;
//                    flX_d->at(i, j, k, HY_XMOM_FLUX_C)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELX_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELX_VAR_C)
//                                                           + sR*sL*(   U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELX_VAR_C)
//                                                                     - U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELX_VAR_C)) )/sRsL;
//                    flX_d->at(i, j, k, HY_XMOM_FLUX_C) += (  sR * U_d->at(iL, j, k, PRES_VAR_C)
//                                                           - sL * U_d->at(iR, j, k, PRES_VAR_C) ) /sRsL;
//                    flX_d->at(i, j, k, HY_YMOM_FLUX_C)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELY_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELY_VAR_C)
//                                                           + sR*sL*(   U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELY_VAR_C)
//                                                                     - U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELY_VAR_C)) )/sRsL;
//                    flX_d->at(i, j, k, HY_ZMOM_FLUX_C)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELZ_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELZ_VAR_C)
//                                                           + sR*sL*(   U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, VELZ_VAR_C)
//                                                                     - U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, VELZ_VAR_C)) )/sRsL;
//                    flX_d->at(i, j, k, HY_ENER_FLUX_C)  = (  sR * vL * U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, ENER_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, ENER_VAR_C)
//                                                           + sR*sL*(   U_d->at(iR, j, k, DENS_VAR_C) * U_d->at(iR, j, k, ENER_VAR_C)
//                                                                     - U_d->at(iL, j, k, DENS_VAR_C) * U_d->at(iL, j, k, ENER_VAR_C)) )/sRsL;
//                    flX_d->at(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * U_d->at(iL, j, k, PRES_VAR_C)
//                                                           - sL * vR * U_d->at(iR, j, k, PRES_VAR_C)) / sRsL;
//                }
//
//                flX_d->at(i, j, k, HY_DENS_FLUX_C) *= dtdx;
//                flX_d->at(i, j, k, HY_XMOM_FLUX_C) *= dtdx;
//                flX_d->at(i, j, k, HY_YMOM_FLUX_C) *= dtdx;
//                flX_d->at(i, j, k, HY_ZMOM_FLUX_C) *= dtdx;
//                flX_d->at(i, j, k, HY_ENER_FLUX_C) *= dtdx;
            }
        }
    }
}

