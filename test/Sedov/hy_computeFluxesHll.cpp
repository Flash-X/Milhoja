#include "Hydro.h"

#include <cmath>
#include <algorithm>

#include "Flash.h"

void hy::computeFluxesHll(const orchestration::Real dt,
                          const orchestration::IntVect& lo,
                          const orchestration::IntVect& hi,
                          const orchestration::RealVect& deltas,
                          const orchestration::FArray4D& Uin,
                          orchestration::FArray4D& flX,
                          orchestration::FArray4D& flY,
                          orchestration::FArray4D& flZ,
                          orchestration::FArray3D& auxC) {
    using namespace orchestration;

//  if (hy_fluxCorrect) then
//     call Driver_abortFlash("hy_hllUnsplit: flux correction is not implemented!")
//  end if
//
//  if (hy_useGravity) then
//     call Driver_abortFlash("hy_hllUnsplit: support for gravity not implemented!")
//  end if
//
//  if (.NOT.hy_updateHydroFluxes) then
//     return
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
//#if NDIM > 1
//  if (tileLimits(HIGH,JAXIS) == ubound(Uin,iX+1)-NGUARD) iLastY = 1
//#endif
//#if NDIM > 2
//  if (tileLimits(HIGH,KAXIS) == ubound(Uin,iX+2)-NGUARD) iLastZ = 1
//#endif

    //************************************************************************
    // Calculate Riemann (interface) states

    // calculate sound speed
    // TODO: Should the limits be the bounds of auxC?  Does that help with
    //       the potential tiling issue?
    for         (int k=lo.K()-K3D; k<=hi.K()+K3D; ++k) {
        for     (int j=lo.J()-K2D; j<=hi.J()+K2D; ++j) {
            for (int i=lo.I()-K1D; i<=hi.I()+K1D; ++i) {
                auxC(i, j, k) = sqrt(  Uin(i, j, k, GAMC_VAR_C)
                                     * Uin(i, j, k, PRES_VAR_C)
                                     / Uin(i, j, k, DENS_VAR_C) );
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
    for         (int k=lo.K(); k<=hi.K();     ++k) {
        for     (int j=lo.J(); j<=hi.J();     ++j) {
            for (int i=lo.I(); i<=hi.I()+K1D; ++i) {
                sL = std::min(Uin(i-1, j, k, VELX_VAR_C) - auxC(i-1, j, k),
                              Uin(i,   j, k, VELX_VAR_C) - auxC(i,   j, k));
                sR = std::max(Uin(i-1, j, k, VELX_VAR_C) + auxC(i-1, j, k),
                              Uin(i,   j, k, VELX_VAR_C) + auxC(i,   j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i-1, j, k, VELX_VAR_C);
                    is = i - 1;
                    iL = i - 1;
                    iR = i - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELX_VAR_C);
                    is = i;
                    iL = i;
                    iR = i;
                } else {
                    vn = 0.5_wp * (  Uin(i-1, j, k, VELX_VAR_C)
                                   + Uin(i,   j, k, VELX_VAR_C));
                    is = i;
                    iL = i-1;
                    iR = i;
                    if (vn > 0.0) {
                        --is;
                    }
                }

                vL = Uin(iL, j, k, VELX_VAR_C);
                vR = Uin(iR, j, k, VELX_VAR_C);
                if (iL == iR) {
                    flX(i, j, k, HY_DENS_FLUX_C) =   vn * Uin(is, j, k, DENS_VAR_C);
                    flX(i, j, k, HY_XMOM_FLUX_C) =   vn * Uin(is, j, k, DENS_VAR_C)
                                                        * Uin(is, j, k, VELX_VAR_C)
                                                   +      Uin(is, j, k, PRES_VAR_C);
                    flX(i, j, k, HY_YMOM_FLUX_C) =   vn * Uin(is, j, k, DENS_VAR_C)
                                                        * Uin(is, j, k, VELY_VAR_C);
                    flX(i, j, k, HY_ZMOM_FLUX_C) =   vn * Uin(is, j, k, DENS_VAR_C)
                                                        * Uin(is, j, k, VELZ_VAR_C);
                    flX(i, j, k, HY_ENER_FLUX_C) =   vn * Uin(is, j, k, DENS_VAR_C)
                                                        * Uin(is, j, k, ENER_VAR_C)
                                                   + vn * Uin(is, j, k, PRES_VAR_C);
                } else {
                    flX(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * Uin(iL, j, k, DENS_VAR_C)
                                                    - sL * vR * Uin(iR, j, k, DENS_VAR_C)
                                                    + sR*sL*(   Uin(iR, j, k, DENS_VAR_C)
                                                              - Uin(iL, j, k, DENS_VAR_C)) ) / sRsL;
                    flX(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELX_VAR_C)
                                                    - sL * vR * Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELX_VAR_C)
                                                    + sR*sL*(   Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELX_VAR_C)
                                                              - Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELX_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_XMOM_FLUX_C) += (  sR * Uin(iL, j, k, PRES_VAR_C)
                                                     - sL * Uin(iR, j, k, PRES_VAR_C) ) /sRsL;
                    flX(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELY_VAR_C)
                                                    - sL * vR * Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELY_VAR_C)
                                                    + sR*sL*(   Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELY_VAR_C)
                                                              - Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELY_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELZ_VAR_C)
                                                    - sL * vR * Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELZ_VAR_C)
                                                    + sR*sL*(   Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, VELZ_VAR_C)
                                                              - Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, VELZ_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, ENER_VAR_C)
                                                    - sL * vR * Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, ENER_VAR_C)
                                                    + sR*sL*(   Uin(iR, j, k, DENS_VAR_C) * Uin(iR, j, k, ENER_VAR_C)
                                                              - Uin(iL, j, k, DENS_VAR_C) * Uin(iL, j, k, ENER_VAR_C)) )/sRsL;
                    flX(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * Uin(iL, j, k, PRES_VAR_C)
                                                     - sL * vR * Uin(iR, j, k, PRES_VAR_C)) / sRsL;
                }

                flX(i, j, k, HY_DENS_FLUX_C) *= dtdx;
                flX(i, j, k, HY_XMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_YMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_ZMOM_FLUX_C) *= dtdx;
                flX(i, j, k, HY_ENER_FLUX_C) *= dtdx;
            }
        }
    }

#if NDIM > 1
    int     js = 0;
    int     jL = 0;
    int     jR = 0;
    Real    dtdy = dt / deltas.J();
    for         (int k=lo.K(); k<=hi.K();     ++k) {
        for     (int j=lo.J(); j<=hi.J()+K2D; ++j) {
            for (int i=lo.I(); i<=hi.I();     ++i) {
                sL = std::min(Uin(i, j-1, k, VELY_VAR_C) - auxC(i, j-1, k),
                              Uin(i, j,   k, VELY_VAR_C) - auxC(i, j,   k));
                sR = std::max(Uin(i, j-1, k, VELY_VAR_C) + auxC(i, j-1, k),
                              Uin(i, j,   k, VELY_VAR_C) + auxC(i, j,   k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i, j-1, k, VELY_VAR_C);
                    js = j - 1;
                    jL = j - 1;
                    jR = j - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELY_VAR_C);
                    js = j;
                    jL = j;
                    jR = j;
                } else {
                    vn = 0.5_wp * (Uin(i, j-1, k, VELY_VAR_C) + Uin(i, j, k, VELY_VAR_C));
                    js = j;
                    jL = j - 1;
                    jR = j;
                    if (vn > 0.0) {
                        --js;
                    }
                }

                vL = Uin(i, jL, k, VELY_VAR_C);
                vR = Uin(i, jR, k, VELY_VAR_C);
                if (jL == jR) {
                    flY(i, j, k, HY_DENS_FLUX_C) =   vn * Uin(i, js, k, DENS_VAR_C);
                    flY(i, j, k, HY_XMOM_FLUX_C) =   vn * Uin(i, js, k, DENS_VAR_C)
                                                        * Uin(i, js, k, VELX_VAR_C);
                    flY(i, j, k, HY_YMOM_FLUX_C) =   vn * Uin(i, js, k, DENS_VAR_C)
                                                        * Uin(i, js, k, VELY_VAR_C)
                                                   +      Uin(i, js, k, PRES_VAR_C);
                    flY(i, j, k, HY_ZMOM_FLUX_C) =   vn * Uin(i, js, k, DENS_VAR_C)
                                                        * Uin(i, js, k, VELZ_VAR_C);
                    flY(i, j, k, HY_ENER_FLUX_C) =   vn * Uin(i, js, k, DENS_VAR_C)
                                                        * Uin(i, js, k, ENER_VAR_C)
                                                   + vn * Uin(i,js,k,PRES_VAR_C);
                } else {
                    flY(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * Uin(i, jL, k, DENS_VAR_C)
                                                    - sL * vR * Uin(i, jR, k, DENS_VAR_C)
                                                    + sR*sL*(   Uin(i, jR, k, DENS_VAR_C)
                                                             -  Uin(i, jL, k, DENS_VAR_C))) /sRsL;
                    flY(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELX_VAR_C)
                                                    - sL * vR * Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELX_VAR_C)
                                                    + sR*sL*(   Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELX_VAR_C)
                                                             -  Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELX_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELY_VAR_C)
                                                    - sL * vR * Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELY_VAR_C)
                                                    + sR*sL*(   Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELY_VAR_C)
                                                             -  Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELY_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_YMOM_FLUX_C) +=  (  sR * Uin(i, jL, k, PRES_VAR_C)
                                                      - sL * Uin(i, jR, k, PRES_VAR_C) ) / sRsL;
                    flY(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELZ_VAR_C)
                                                    - sL * vR * Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELZ_VAR_C)
                                                    + sR*sL*(   Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, VELZ_VAR_C)
                                                             -  Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, VELZ_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, ENER_VAR_C)
                                                    - sL * vR * Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, ENER_VAR_C)
                                                    + sR*sL*(   Uin(i, jR, k, DENS_VAR_C) * Uin(i, jR, k, ENER_VAR_C)
                                                             -  Uin(i, jL, k, DENS_VAR_C) * Uin(i, jL, k, ENER_VAR_C)) ) /sRsL;
                    flY(i, j, k, HY_ENER_FLUX_C) +=  (  sR * vL * Uin(i, jL, k, PRES_VAR_C)
                                                      - sL * vR * Uin(i, jR, k, PRES_VAR_C) ) /sRsL;
                }

                flY(i, j, k, HY_DENS_FLUX_C) *= dtdy; 
                flY(i, j, k, HY_XMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_YMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_ZMOM_FLUX_C) *= dtdy;
                flY(i, j, k, HY_ENER_FLUX_C) *= dtdy;
            }
        }
    }
#endif

#if NDIM > 2
    int     ks = 0;
    int     kL = 0;
    int     kR = 0;
    Real    dtdz = dt / deltas.K();
    for         (int k=lo.K(); k<=hi.K()+K3D; ++k) {
        for     (int j=lo.J(); j<=hi.J();     ++j) {
            for (int i=lo.I(); i<=hi.I();     ++i) {
                sL = std::min(Uin(i, j, k-1, VELZ_VAR_C) - auxC(i, j, k-1),
                              Uin(i, j, k,   VELZ_VAR_C) - auxC(i, j, k));
                sR = std::max(Uin(i, j, k-1, VELZ_VAR_C) + auxC(i, j, k-1),
                              Uin(i, j, k,   VELZ_VAR_C) + auxC(i, j, k));
                sRsL = sR - sL;
                if (sL > 0.0) {
                    vn = Uin(i, j, k-1, VELZ_VAR_C);
                    ks = k - 1;
                    kL = k - 1;
                    kR = k - 1;
                } else if (sR < 0.0) {
                    vn = Uin(i, j, k, VELZ_VAR_C);
                    ks = k;
                    kL = k;
                    kR = k;
                } else {
                    vn = 0.5_wp * (  Uin(i, j, k-1, VELZ_VAR_C)
                                   + Uin(i, j, k,   VELZ_VAR_C));
                    ks = k;
                    kL = k-1;
                    kR = k;
                    if (vn > 0.0) {
                      --ks;
                    }
                }

                vL = Uin(i, j, kL, VELZ_VAR_C);
                vR = Uin(i, j, kR, VELZ_VAR_C);
                if (kL == kR) {
                    flZ(i, j, k, HY_DENS_FLUX_C) =   vn * Uin(i, j, ks, DENS_VAR_C);
                    flZ(i, j, k, HY_XMOM_FLUX_C) =   vn * Uin(i, j, ks, DENS_VAR_C)
                                                        * Uin(i, j, ks, VELX_VAR_C);
                    flZ(i, j, k, HY_YMOM_FLUX_C) =   vn * Uin(i, j, ks, DENS_VAR_C)
                                                        * Uin(i, j, ks, VELY_VAR_C);
                    flZ(i, j, k, HY_ZMOM_FLUX_C) =   vn * Uin(i, j, ks, DENS_VAR_C)
                                                        * Uin(i, j, ks, VELZ_VAR_C)
                                                   +      Uin(i, j, ks, PRES_VAR_C);
                    flZ(i, j, k, HY_ENER_FLUX_C) =   vn * Uin(i, j, ks, DENS_VAR_C)
                                                        * Uin(i, j, ks, ENER_VAR_C)
                                                   + vn * Uin(i, j, ks, PRES_VAR_C);
                } else {
                    flZ(i, j, k, HY_DENS_FLUX_C) = (  sR * vL * Uin(i, j, kL, DENS_VAR_C)
                                                    - sL * vR * Uin(i, j, kR, DENS_VAR_C)
                                                    + sR*sL*(   Uin(i, j, kR, DENS_VAR_C)
                                                             -  Uin(i, j, kL, DENS_VAR_C))) /sRsL;
                    flZ(i, j, k, HY_XMOM_FLUX_C) = (  sR * vL * Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELX_VAR_C)
                                                    - sL * vR * Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELX_VAR_C)
                                                    + sR*sL*(   Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELX_VAR_C)
                                                             -  Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELX_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_YMOM_FLUX_C) = (  sR * vL * Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELY_VAR_C)
                                                    - sL * vR * Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELY_VAR_C)
                                                    + sR*sL*(   Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELY_VAR_C)
                                                             -  Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELY_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX_C) = (  sR * vL * Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELZ_VAR_C)
                                                    - sL * vR * Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELZ_VAR_C)
                                                    + sR*sL*(   Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, VELZ_VAR_C)
                                                             -  Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, VELZ_VAR_C)) ) /sRsL;
                    flZ(i, j, k, HY_ZMOM_FLUX_C) += (  sR * Uin(i, j, kL, PRES_VAR_C)
                                                     - sL * Uin(i, j, kR, PRES_VAR_C) ) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX_C) = (  sR * vL * Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, ENER_VAR_C)
                                                    - sL * vR * Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, ENER_VAR_C)
                                                    + sR*sL*(   Uin(i, j, kR, DENS_VAR_C) * Uin(i, j, kR, ENER_VAR_C)
                                                             -  Uin(i, j, kL, DENS_VAR_C) * Uin(i, j, kL, ENER_VAR_C))) /sRsL;
                    flZ(i, j, k, HY_ENER_FLUX_C) += (  sR * vL * Uin(i, j, kL, PRES_VAR_C)
                                                     - sL * vR * Uin(i, j, kR, PRES_VAR_C) ) /sRsL;
                }

                flZ(i, j, k, HY_DENS_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_XMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_YMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_ZMOM_FLUX_C) *= dtdz;
                flZ(i, j, k, HY_ENER_FLUX_C) *= dtdz;
            }
        }
    }
#endif
}

