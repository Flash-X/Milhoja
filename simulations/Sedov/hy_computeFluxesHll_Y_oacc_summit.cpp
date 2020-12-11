#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

//#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll_Y_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flY_d,
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
//    int     js = 0;
//    int     jL = 0;
//    int     jR = 0;
    Real    dtdy = *dt_d / deltas_d->J();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e;     ++k) {
        for     (int j=j_s; j<=j_e+K2D; ++j) {
            for (int i=i_s; i<=i_e;     ++i) {
                flY_d->at(i, j, k, HY_YMOM_FLUX_C) = dtdy;
//                sL = std::min(U_d->at(i, j-1, k, VELY_VAR_C) - auxC_d->at(i, j-1, k, 0),
//                              U_d->at(i, j,   k, VELY_VAR_C) - auxC_d->at(i, j,   k, 0));
//                sR = std::max(U_d->at(i, j-1, k, VELY_VAR_C) + auxC_d->at(i, j-1, k, 0),
//                              U_d->at(i, j,   k, VELY_VAR_C) + auxC_d->at(i, j,   k, 0));
//                sRsL = sR - sL;
//                if (sL > 0.0) {
//                    vn = U_d->at(i, j-1, k, VELY_VAR_C);
//                    js = j - 1;
//                    jL = j - 1;
//                    jR = j - 1;
//                } else if (sR < 0.0) {
//                    vn = U_d->at(i, j, k, VELY_VAR_C);
//                    js = j;
//                    jL = j;
//                    jR = j;
//                } else {
//                    vn = 0.5_wp * (  U_d->at(i, j-1, k, VELY_VAR_C)
//                                   + U_d->at(i, j,   k, VELY_VAR_C));
//                    js = j;
//                    jL = j - 1;
//                    jR = j;
//                    if (vn > 0.0) {
//                        --js;
//                    }
//                }
//
//                vL = U_d->at(i, jL, k, VELY_VAR_C);
//                vR = U_d->at(i, jR, k, VELY_VAR_C);
//                if (jL == jR) {
//                    flY_d->at(i, j, k, HY_DENS_FLUX_C) =   vn * U_d->at(i, js, k, DENS_VAR_C);
//                    flY_d->at(i, j, k, HY_XMOM_FLUX_C) =   vn * U_d->at(i, js, k, DENS_VAR_C)
//                                                              * U_d->at(i, js, k, VELX_VAR_C);
//                    flY_d->at(i, j, k, HY_YMOM_FLUX_C) =   vn * U_d->at(i, js, k, DENS_VAR_C)
//                                                              * U_d->at(i, js, k, VELY_VAR_C)
//                                                         +      U_d->at(i, js, k, PRES_VAR_C);
//                    flY_d->at(i, j, k, HY_ZMOM_FLUX_C) =   vn * U_d->at(i, js, k, DENS_VAR_C)
//                                                              * U_d->at(i, js, k, VELZ_VAR_C);
//                    flY_d->at(i, j, k, HY_ENER_FLUX_C) =   vn * U_d->at(i, js, k, DENS_VAR_C)
//                                                              * U_d->at(i, js, k, ENER_VAR_C)
//                                                         + vn * U_d->at(i,js,k,PRES_VAR_C);
//                } else {
//                    flY_d->at(i, j, k, HY_DENS_FLUX_C)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, DENS_VAR_C)
//                                                           + sR*sL*(   U_d->at(i, jR, k, DENS_VAR_C)
//                                                                    -  U_d->at(i, jL, k, DENS_VAR_C))) /sRsL;
//                    flY_d->at(i, j, k, HY_XMOM_FLUX_C)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELX_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELX_VAR_C)
//                                                           + sR*sL*(   U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELX_VAR_C)
//                                                                    -  U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELX_VAR_C)) ) /sRsL;
//                    flY_d->at(i, j, k, HY_YMOM_FLUX_C)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELY_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELY_VAR_C)
//                                                           + sR*sL*(   U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELY_VAR_C)
//                                                                    -  U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELY_VAR_C)) ) /sRsL;
//                    flY_d->at(i, j, k, HY_YMOM_FLUX_C) += (  sR * U_d->at(i, jL, k, PRES_VAR_C)
//                                                           - sL * U_d->at(i, jR, k, PRES_VAR_C) ) / sRsL;
//                    flY_d->at(i, j, k, HY_ZMOM_FLUX_C)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELZ_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELZ_VAR_C)
//                                                           + sR*sL*(   U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, VELZ_VAR_C)
//                                                                    -  U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, VELZ_VAR_C)) ) /sRsL;
//                    flY_d->at(i, j, k, HY_ENER_FLUX_C)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, ENER_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, ENER_VAR_C)
//                                                           + sR*sL*(   U_d->at(i, jR, k, DENS_VAR_C) * U_d->at(i, jR, k, ENER_VAR_C)
//                                                                    -  U_d->at(i, jL, k, DENS_VAR_C) * U_d->at(i, jL, k, ENER_VAR_C)) ) /sRsL;
//                    flY_d->at(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * U_d->at(i, jL, k, PRES_VAR_C)
//                                                           - sL * vR * U_d->at(i, jR, k, PRES_VAR_C) ) /sRsL;
//                }
//
//                flY_d->at(i, j, k, HY_DENS_FLUX_C) *= dtdy; 
//                flY_d->at(i, j, k, HY_XMOM_FLUX_C) *= dtdy;
//                flY_d->at(i, j, k, HY_YMOM_FLUX_C) *= dtdy;
//                flY_d->at(i, j, k, HY_ZMOM_FLUX_C) *= dtdy;
//                flY_d->at(i, j, k, HY_ENER_FLUX_C) *= dtdy;
            }
        }
    }
}

