#include "Hydro.h"

#include <algorithm>

#include <Milhoja.h>

#include "Sedov.h"

#ifndef MILHOJA_ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void hy::computeFluxesHll_Y_oacc_summit(const milhoja::Real* dt_d,
                                        const milhoja::IntVect* lo_d,
                                        const milhoja::IntVect* hi_d,
                                        const milhoja::RealVect* deltas_d,
                                        const milhoja::FArray4D* U_d,
                                        milhoja::FArray4D* flY_d,
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
    int     js = 0;
    int     jL = 0;
    int     jR = 0;
    Real    dtdy = *dt_d / deltas_d->J();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e;             ++k) {
        for     (int j=j_s; j<=j_e+MILHOJA_K2D; ++j) {
            for (int i=i_s; i<=i_e;             ++i) {
                sL = std::min(U_d->at(i, j-1, k, VELY_VAR) - auxC_d->at(i, j-1, k, 0),
                              U_d->at(i, j,   k, VELY_VAR) - auxC_d->at(i, j,   k, 0));
                sR = std::max(U_d->at(i, j-1, k, VELY_VAR) + auxC_d->at(i, j-1, k, 0),
                              U_d->at(i, j,   k, VELY_VAR) + auxC_d->at(i, j,   k, 0));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = U_d->at(i, j-1, k, VELY_VAR);
                    js = j - 1;
                    jL = j - 1;
                    jR = j - 1;
                } else if (sR < 0.0) {
                    vn = U_d->at(i, j, k, VELY_VAR);
                    js = j;
                    jL = j;
                    jR = j;
                } else {
                    vn = 0.5_wp * (  U_d->at(i, j-1, k, VELY_VAR)
                                   + U_d->at(i, j,   k, VELY_VAR));
                    js = j;
                    jL = j - 1;
                    jR = j;
                    if (vn > 0.0) {
                        --js;
                    }
                }

                vL = U_d->at(i, jL, k, VELY_VAR);
                vR = U_d->at(i, jR, k, VELY_VAR);
                if (jL == jR) {
                    flY_d->at(i, j, k, HY_DENS_FLUX) =   vn * U_d->at(i, js, k, DENS_VAR);
                    flY_d->at(i, j, k, HY_XMOM_FLUX) =   vn * U_d->at(i, js, k, DENS_VAR)
                                                            * U_d->at(i, js, k, VELX_VAR);
                    flY_d->at(i, j, k, HY_YMOM_FLUX) =   vn * U_d->at(i, js, k, DENS_VAR)
                                                            * U_d->at(i, js, k, VELY_VAR)
                                                       +      U_d->at(i, js, k, PRES_VAR);
                    flY_d->at(i, j, k, HY_ZMOM_FLUX) =   vn * U_d->at(i, js, k, DENS_VAR)
                                                            * U_d->at(i, js, k, VELZ_VAR);
                    flY_d->at(i, j, k, HY_ENER_FLUX) =   vn * U_d->at(i, js, k, DENS_VAR)
                                                            * U_d->at(i, js, k, ENER_VAR)
                                                       + vn * U_d->at(i,js,k,PRES_VAR);
                } else {
                    flY_d->at(i, j, k, HY_DENS_FLUX)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, DENS_VAR)
                                                         + sR*sL*(   U_d->at(i, jR, k, DENS_VAR)
                                                                  -  U_d->at(i, jL, k, DENS_VAR))) /sRsL;
                    flY_d->at(i, j, k, HY_XMOM_FLUX)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELX_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELX_VAR)
                                                         + sR*sL*(   U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELX_VAR)
                                                                  -  U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELX_VAR)) ) /sRsL;
                    flY_d->at(i, j, k, HY_YMOM_FLUX)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELY_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELY_VAR)
                                                         + sR*sL*(   U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELY_VAR)
                                                                  -  U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELY_VAR)) ) /sRsL;
                    flY_d->at(i, j, k, HY_YMOM_FLUX) += (  sR * U_d->at(i, jL, k, PRES_VAR)
                                                         - sL * U_d->at(i, jR, k, PRES_VAR) ) / sRsL;
                    flY_d->at(i, j, k, HY_ZMOM_FLUX)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELZ_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELZ_VAR)
                                                         + sR*sL*(   U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, VELZ_VAR)
                                                                  -  U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, VELZ_VAR)) ) /sRsL;
                    flY_d->at(i, j, k, HY_ENER_FLUX)  = (  sR * vL * U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, ENER_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, ENER_VAR)
                                                         + sR*sL*(   U_d->at(i, jR, k, DENS_VAR) * U_d->at(i, jR, k, ENER_VAR)
                                                                  -  U_d->at(i, jL, k, DENS_VAR) * U_d->at(i, jL, k, ENER_VAR)) ) /sRsL;
                    flY_d->at(i, j, k, HY_ENER_FLUX) += (  sR * vL * U_d->at(i, jL, k, PRES_VAR)
                                                         - sL * vR * U_d->at(i, jR, k, PRES_VAR) ) /sRsL;
                }

                flY_d->at(i, j, k, HY_DENS_FLUX) *= dtdy; 
                flY_d->at(i, j, k, HY_XMOM_FLUX) *= dtdy;
                flY_d->at(i, j, k, HY_YMOM_FLUX) *= dtdy;
                flY_d->at(i, j, k, HY_ZMOM_FLUX) *= dtdy;
                flY_d->at(i, j, k, HY_ENER_FLUX) *= dtdy;
            }
        }
    }
}

