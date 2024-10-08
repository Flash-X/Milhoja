#include "Hydro.h"

#include <cmath>
#include <algorithm>

#include "Sedov.h"

void hy::computeFluxesHll(const milhoja::Real dt,
                          const milhoja::IntVect& lo,
                          const milhoja::IntVect& hi,
                          const milhoja::RealVect& deltas,
                          const milhoja::FArray4D& Uin,
                          LIST_NDIM(milhoja::FArray4D& flX, milhoja::FArray4D& flY, milhoja::FArray4D& flZ),
                          milhoja::FArray3D& auxC) {
    //$milhoja  "U": {
    //$milhoja&    "R": [GAMC_VAR,
    //$milhoja&          VELX_VAR, VELY_VAR, VELZ_VAR,
    //$milhoja&          DENS_VAR, PRES_VAR, ENER_VAR],
    //$milhoja& },
    //$milhoja& "flX": {
    //$milhoja&    "W": [HY_XMOM_FLUX, HY_YMOM_FLUX, HY_ZMOM_FLUX,
    //$milhoja&          HY_DENS_FLUX, HY_ENER_FLUX]
    //$milhoja& }

    using namespace milhoja;

//  if (hy_fluxCorrect) then
//     call Driver_abortFlash("hy_hllUnsplit: flux correction is not implemented!")
//  end if

    // For each relevant direction We set indicators to determine whether
    // fluxes on the last face (to the right of the last cell in the
    // cell-centered range given in tileLimits) need to be computed or
    // not. They need to be computed only if the face is on the boundary
    // (between the last interior in the first guard cell). If proper
    // tiling is in effect, we want to avoid computing the same flux
    // twice if the last face of some tile coincides with the first face
    // of the next tile.
    // The following logic assumes that Uin is sized such that the high
    // bound of its relevant indices is exactly NGUARD cells to the
    // right of the last interior cell.
    // TODO: These routines should not have any notion of tiles/blocks.
    // Therefore, this information should come from the calling code.
    // How to do this?
//  if (tileLimits(HIGH,IAXIS) == ubound(Uin,iX  )-NGUARD) iLastX = 1
//#if (MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)
//  if (tileLimits(HIGH,JAXIS) == ubound(Uin,iX+1)-NGUARD) iLastY = 1
//#endif
//#if MILHOJA_NDIM == 3
//  if (tileLimits(HIGH,KAXIS) == ubound(Uin,iX+2)-NGUARD) iLastZ = 1
//#endif

    //************************************************************************
    // Calculate Riemann (interface) states

    // calculate sound speed
    // TODO: Should the limits be the bounds of auxC?  Does that help with
    //       the potential tiling issue?
    for         (int k=lo.K()-MILHOJA_K3D; k<=hi.K()+MILHOJA_K3D; ++k) {
        for     (int j=lo.J()-MILHOJA_K2D; j<=hi.J()+MILHOJA_K2D; ++j) {
            for (int i=lo.I()-MILHOJA_K1D; i<=hi.I()+MILHOJA_K1D; ++i) {
                auxC(i, j, k) = sqrt(  Uin(i, j, k, GAMC_VAR)
                                     * Uin(i, j, k, PRES_VAR)
                                     / Uin(i, j, k, DENS_VAR) );
            }
        }
    }

    //************************************************************************
    // Calculate Godunov fluxes

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
    for         (int k=lo.K(); k<=hi.K();             ++k) {
        for     (int j=lo.J(); j<=hi.J();             ++j) {
            for (int i=lo.I(); i<=hi.I()+MILHOJA_K1D; ++i) {
                sL = std::min(Uin(i-1, j, k, VELX_VAR) - auxC(i-1, j, k),
                              Uin(i,   j, k, VELX_VAR) - auxC(i,   j, k));
                sR = std::max(Uin(i-1, j, k, VELX_VAR) + auxC(i-1, j, k),
                              Uin(i,   j, k, VELX_VAR) + auxC(i,   j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i-1, j, k, VELX_VAR);
                    is = i - 1;
                    iL = i - 1;
                    iR = i - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELX_VAR);
                    is = i;
                    iL = i;
                    iR = i;
                } else {
                    vn = 0.5_wp * (  Uin(i-1, j, k, VELX_VAR)
                                   + Uin(i,   j, k, VELX_VAR));
                    is = i;
                    iL = i-1;
                    iR = i;
                    if (vn > 0.0) {
                        --is;
                    }
                }

                vL = Uin(iL, j, k, VELX_VAR);
                vR = Uin(iR, j, k, VELX_VAR);
                if (iL == iR) {
                    flX(i, j, k, HY_DENS_FLUX) =   vn * Uin(is, j, k, DENS_VAR);
                    flX(i, j, k, HY_XMOM_FLUX) =   vn * Uin(is, j, k, DENS_VAR)
                                                      * Uin(is, j, k, VELX_VAR)
                                                 +      Uin(is, j, k, PRES_VAR);
                    flX(i, j, k, HY_YMOM_FLUX) =   vn * Uin(is, j, k, DENS_VAR)
                                                      * Uin(is, j, k, VELY_VAR);
                    flX(i, j, k, HY_ZMOM_FLUX) =   vn * Uin(is, j, k, DENS_VAR)
                                                      * Uin(is, j, k, VELZ_VAR);
                    flX(i, j, k, HY_ENER_FLUX) =   vn * Uin(is, j, k, DENS_VAR)
                                                      * Uin(is, j, k, ENER_VAR)
                                                 + vn * Uin(is, j, k, PRES_VAR);
                } else {
                    flX(i, j, k, HY_DENS_FLUX) = (  sR * vL * Uin(iL, j, k, DENS_VAR)
                                                  - sL * vR * Uin(iR, j, k, DENS_VAR)
                                                  + sR*sL*(   Uin(iR, j, k, DENS_VAR)
                                                            - Uin(iL, j, k, DENS_VAR)) ) / sRsL;
                    flX(i, j, k, HY_XMOM_FLUX) = (  sR * vL * Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELX_VAR)
                                                  - sL * vR * Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELX_VAR)
                                                  + sR*sL*(   Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELX_VAR)
                                                            - Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELX_VAR)) )/sRsL;
                    flX(i, j, k, HY_XMOM_FLUX) += (  sR * Uin(iL, j, k, PRES_VAR)
                                                   - sL * Uin(iR, j, k, PRES_VAR) ) /sRsL;
                    flX(i, j, k, HY_YMOM_FLUX) = (  sR * vL * Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELY_VAR)
                                                  - sL * vR * Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELY_VAR)
                                                  + sR*sL*(   Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELY_VAR)
                                                            - Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELY_VAR)) )/sRsL;
                    flX(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELZ_VAR)
                                                  - sL * vR * Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELZ_VAR)
                                                  + sR*sL*(   Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, VELZ_VAR)
                                                            - Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, VELZ_VAR)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX) = (  sR * vL * Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, ENER_VAR)
                                                  - sL * vR * Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, ENER_VAR)
                                                  + sR*sL*(   Uin(iR, j, k, DENS_VAR) * Uin(iR, j, k, ENER_VAR)
                                                            - Uin(iL, j, k, DENS_VAR) * Uin(iL, j, k, ENER_VAR)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX) += (  sR * vL * Uin(iL, j, k, PRES_VAR)
                                                   - sL * vR * Uin(iR, j, k, PRES_VAR)) / sRsL;
                }

                flX(i, j, k, HY_DENS_FLUX) *= dtdx;
                flX(i, j, k, HY_XMOM_FLUX) *= dtdx;
                flX(i, j, k, HY_YMOM_FLUX) *= dtdx;
                flX(i, j, k, HY_ZMOM_FLUX) *= dtdx;
                flX(i, j, k, HY_ENER_FLUX) *= dtdx;
            }
        }
    }

#if (MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)
    int     js = 0;
    int     jL = 0;
    int     jR = 0;
    Real    dtdy = dt / deltas.J();
    for         (int k=lo.K(); k<=hi.K();             ++k) {
        for     (int j=lo.J(); j<=hi.J()+MILHOJA_K2D; ++j) {
            for (int i=lo.I(); i<=hi.I();             ++i) {
                sL = std::min(Uin(i, j-1, k, VELY_VAR) - auxC(i, j-1, k),
                              Uin(i, j,   k, VELY_VAR) - auxC(i, j,   k));
                sR = std::max(Uin(i, j-1, k, VELY_VAR) + auxC(i, j-1, k),
                              Uin(i, j,   k, VELY_VAR) + auxC(i, j,   k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i, j-1, k, VELY_VAR);
                    js = j - 1;
                    jL = j - 1;
                    jR = j - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELY_VAR);
                    js = j;
                    jL = j;
                    jR = j;
                } else {
                    vn = 0.5_wp * (Uin(i, j-1, k, VELY_VAR) + Uin(i, j, k, VELY_VAR));
                    js = j;
                    jL = j - 1;
                    jR = j;
                    if (vn > 0.0) {
                        --js;
                    }
                }

                vL = Uin(i, jL, k, VELY_VAR);
                vR = Uin(i, jR, k, VELY_VAR);
                if (jL == jR) {
                    flY(i, j, k, HY_DENS_FLUX) =   vn * Uin(i, js, k, DENS_VAR);
                    flY(i, j, k, HY_XMOM_FLUX) =   vn * Uin(i, js, k, DENS_VAR)
                                                      * Uin(i, js, k, VELX_VAR);
                    flY(i, j, k, HY_YMOM_FLUX) =   vn * Uin(i, js, k, DENS_VAR)
                                                      * Uin(i, js, k, VELY_VAR)
                                                 +      Uin(i, js, k, PRES_VAR);
                    flY(i, j, k, HY_ZMOM_FLUX) =   vn * Uin(i, js, k, DENS_VAR)
                                                      * Uin(i, js, k, VELZ_VAR);
                    flY(i, j, k, HY_ENER_FLUX) =   vn * Uin(i, js, k, DENS_VAR)
                                                      * Uin(i, js, k, ENER_VAR)
                                                 + vn * Uin(i,js,k,PRES_VAR);
                } else {
                    flY(i, j, k, HY_DENS_FLUX) = (  sR * vL * Uin(i, jL, k, DENS_VAR)
                                                  - sL * vR * Uin(i, jR, k, DENS_VAR)
                                                  + sR*sL*(   Uin(i, jR, k, DENS_VAR)
                                                           -  Uin(i, jL, k, DENS_VAR))) /sRsL;
                    flY(i, j, k, HY_XMOM_FLUX) = (  sR * vL * Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELX_VAR)
                                                  - sL * vR * Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELX_VAR)
                                                  + sR*sL*(   Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELX_VAR)
                                                           -  Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELX_VAR)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX) = (  sR * vL * Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELY_VAR)
                                                  - sL * vR * Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELY_VAR)
                                                  + sR*sL*(   Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELY_VAR)
                                                           -  Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELY_VAR)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX) +=  (  sR * Uin(i, jL, k, PRES_VAR)
                                                    - sL * Uin(i, jR, k, PRES_VAR) ) / sRsL;
                    flY(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELZ_VAR)
                                                  - sL * vR * Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELZ_VAR)
                                                  + sR*sL*(   Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, VELZ_VAR)
                                                           -  Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, VELZ_VAR)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX) = (  sR * vL * Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, ENER_VAR)
                                                  - sL * vR * Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, ENER_VAR)
                                                  + sR*sL*(   Uin(i, jR, k, DENS_VAR) * Uin(i, jR, k, ENER_VAR)
                                                           -  Uin(i, jL, k, DENS_VAR) * Uin(i, jL, k, ENER_VAR)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX) +=  (  sR * vL * Uin(i, jL, k, PRES_VAR)
                                                    - sL * vR * Uin(i, jR, k, PRES_VAR) ) /sRsL;
                }

                flY(i, j, k, HY_DENS_FLUX) *= dtdy; 
                flY(i, j, k, HY_XMOM_FLUX) *= dtdy;
                flY(i, j, k, HY_YMOM_FLUX) *= dtdy;
                flY(i, j, k, HY_ZMOM_FLUX) *= dtdy;
                flY(i, j, k, HY_ENER_FLUX) *= dtdy;
            }
        }
    }
#endif

#if MILHOJA_NDIM == 3
    int     ks = 0;
    int     kL = 0;
    int     kR = 0;
    Real    dtdz = dt / deltas.K();
    for         (int k=lo.K(); k<=hi.K()+MILHOJA_K3D; ++k) {
        for     (int j=lo.J(); j<=hi.J();             ++j) {
            for (int i=lo.I(); i<=hi.I();             ++i) {
                sL = std::min(Uin(i, j, k-1, VELZ_VAR) - auxC(i, j, k-1),
                              Uin(i, j, k,   VELZ_VAR) - auxC(i, j, k));
                sR = std::max(Uin(i, j, k-1, VELZ_VAR) + auxC(i, j, k-1),
                              Uin(i, j, k,   VELZ_VAR) + auxC(i, j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i, j, k-1, VELZ_VAR);
                    ks = k - 1;
                    kL = k - 1;
                    kR = k - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELZ_VAR);
                    ks = k;
                    kL = k;
                    kR = k;
                } else {
                    vn = 0.5_wp * (  Uin(i, j, k-1, VELZ_VAR)
                                   + Uin(i, j, k,   VELZ_VAR));
                    ks = k;
                    kL = k-1;
                    kR = k;
                    if (vn > 0.0) {
                      --ks;
                    }
                }

                vL = Uin(i, j, kL, VELZ_VAR);
                vR = Uin(i, j, kR, VELZ_VAR);
                if (kL == kR) {
                    flZ(i, j, k, HY_DENS_FLUX) =   vn * Uin(i, j, ks, DENS_VAR);
                    flZ(i, j, k, HY_XMOM_FLUX) =   vn * Uin(i, j, ks, DENS_VAR)
                                                      * Uin(i, j, ks, VELX_VAR);
                    flZ(i, j, k, HY_YMOM_FLUX) =   vn * Uin(i, j, ks, DENS_VAR)
                                                      * Uin(i, j, ks, VELY_VAR);
                    flZ(i, j, k, HY_ZMOM_FLUX) =   vn * Uin(i, j, ks, DENS_VAR)
                                                      * Uin(i, j, ks, VELZ_VAR)
                                                 +      Uin(i, j, ks, PRES_VAR);
                    flZ(i, j, k, HY_ENER_FLUX) =   vn * Uin(i, j, ks, DENS_VAR)
                                                      * Uin(i, j, ks, ENER_VAR)
                                                 + vn * Uin(i, j, ks, PRES_VAR);
                } else {
                    flZ(i, j, k, HY_DENS_FLUX) = (  sR * vL * Uin(i, j, kL, DENS_VAR)
                                                  - sL * vR * Uin(i, j, kR, DENS_VAR)
                                                  + sR*sL*(   Uin(i, j, kR, DENS_VAR)
                                                           -  Uin(i, j, kL, DENS_VAR))) /sRsL;
                    flZ(i, j, k, HY_XMOM_FLUX) = (  sR * vL * Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELX_VAR)
                                                  - sL * vR * Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELX_VAR)
                                                  + sR*sL*(   Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELX_VAR)
                                                           -  Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELX_VAR)) ) /sRsL;
                    flZ(i, j, k, HY_YMOM_FLUX) = (  sR * vL * Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELY_VAR)
                                                  - sL * vR * Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELY_VAR)
                                                  + sR*sL*(   Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELY_VAR)
                                                           -  Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELY_VAR)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELZ_VAR)
                                                  - sL * vR * Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELZ_VAR)
                                                  + sR*sL*(   Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, VELZ_VAR)
                                                           -  Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, VELZ_VAR)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX) += (  sR * Uin(i, j, kL, PRES_VAR)
                                                   - sL * Uin(i, j, kR, PRES_VAR) ) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX) = (  sR * vL * Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, ENER_VAR)
                                                  - sL * vR * Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, ENER_VAR)
                                                  + sR*sL*(   Uin(i, j, kR, DENS_VAR) * Uin(i, j, kR, ENER_VAR)
                                                           -  Uin(i, j, kL, DENS_VAR) * Uin(i, j, kL, ENER_VAR))) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX) += (  sR * vL * Uin(i, j, kL, PRES_VAR)
                                                   - sL * vR * Uin(i, j, kR, PRES_VAR) ) /sRsL;
                }

                flZ(i, j, k, HY_DENS_FLUX) *= dtdz;
                flZ(i, j, k, HY_XMOM_FLUX) *= dtdz;
                flZ(i, j, k, HY_YMOM_FLUX) *= dtdz;
                flZ(i, j, k, HY_ZMOM_FLUX) *= dtdz;
                flZ(i, j, k, HY_ENER_FLUX) *= dtdz;
            }
        }
    }
#endif
}

