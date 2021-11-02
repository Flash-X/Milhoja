#include "constants.h"
#include "Flash.h"

subroutine hy_computeFluxesHll_Y_oacc(dt, lo, hi, deltas, U, auxC, flY)
    implicit none

    !$acc routine vector

    real,    intent(IN)  :: dt
    integer, intent(IN)  :: lo(:)
    integer, intent(IN)  :: hi(:)
    real,    intent(IN)  :: deltas(:)
    real,    intent(IN)  :: U(:, :, :, :)
    real,    intent(IN)  :: auxC(:, :, :)
    real,    intent(OUT) :: flY(:, :, :, :)

    real    :: sL
    real    :: sR
    real    :: sRsL
    real    :: vn
    real    :: vL
    real    :: vR
    integer :: js
    integer :: jL
    integer :: jR
    real    :: dtdy

    integer :: i, j, k

    dtdy = dt / deltas(JAXIS)

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)+K2D
            do i = lo(IAXIS), hi(IAXIS)
                sL = MIN(U(i, j-1, k, VELY_VAR) - auxC(i, j-1, k), &
                         U(i, j,   k, VELY_VAR) - auxC(i, j,   k))
                sR = MAX(U(i, j-1, k, VELY_VAR) + auxC(i, j-1, k), &
                         U(i, j,   k, VELY_VAR) + auxC(i, j,   k))
                sRsL = sR - sL
                if (sL > 0.0) then
                    vn = U(i, j-1, k, VELY_VAR)
                    js = j - 1
                    jL = j - 1
                    jR = j - 1
                else if (sR < 0.0) then
                    vn = U(i, j, k, VELY_VAR)
                    js = j
                    jL = j
                    jR = j
                else
                    vn = 0.5 * (  U(i, j-1, k, VELY_VAR)  &
                                + U(i, j,   k, VELY_VAR))
                    js = j
                    jL = j - 1
                    jR = j
                    if (vn > 0.0) then
                        js = js - 1
                    end if
                end if

                vL = U(i, jL, k, VELY_VAR)
                vR = U(i, jR, k, VELY_VAR)
                if (jL == jR) then
                    flY(i, j, k, HY_DENS_FLUX) =   vn * U(i, js, k, DENS_VAR)
                    flY(i, j, k, HY_XMOM_FLUX) =   vn * U(i, js, k, DENS_VAR) &
                                                      * U(i, js, k, VELX_VAR)
                    flY(i, j, k, HY_YMOM_FLUX) =   vn * U(i, js, k, DENS_VAR) &
                                                      * U(i, js, k, VELY_VAR) &
                                                 +      U(i, js, k, PRES_VAR)
                    flY(i, j, k, HY_ZMOM_FLUX) =   vn * U(i, js, k, DENS_VAR) &
                                                      * U(i, js, k, VELZ_VAR)
                    flY(i, j, k, HY_ENER_FLUX) =   vn * U(i, js, k, DENS_VAR) &
                                                      * U(i, js, k, ENER_VAR) &
                                                 + vn * U(i, js, k, PRES_VAR)
                else
                    flY(i, j, k, HY_DENS_FLUX) = (  sR * vL * U(i, jL, k, DENS_VAR) &
                                                  - sL * vR * U(i, jR, k, DENS_VAR) &
                                                  + sR*sL*(   U(i, jR, k, DENS_VAR) &
                                                           -  U(i, jL, k, DENS_VAR))) / sRsL
                    flY(i, j, k, HY_XMOM_FLUX) = (  sR * vL * U(i, jL, k, DENS_VAR) * U(i, jL, k, VELX_VAR) &
                                                  - sL * vR * U(i, jR, k, DENS_VAR) * U(i, jR, k, VELX_VAR) &
                                                  + sR*sL*(   U(i, jR, k, DENS_VAR) * U(i, jR, k, VELX_VAR) &
                                                           -  U(i, jL, k, DENS_VAR) * U(i, jL, k, VELX_VAR)) ) / sRsL
                    flY(i, j, k, HY_YMOM_FLUX) = (  sR * vL * U(i, jL, k, DENS_VAR) * U(i, jL, k, VELY_VAR)  &
                                                  - sL * vR * U(i, jR, k, DENS_VAR) * U(i, jR, k, VELY_VAR)  &
                                                  + sR*sL*(   U(i, jR, k, DENS_VAR) * U(i, jR, k, VELY_VAR)  &
                                                           -  U(i, jL, k, DENS_VAR) * U(i, jL, k, VELY_VAR)) &
                                                  + sR *      U(i, jL, k, PRES_VAR)                          &
                                                  - sL *      U(i, jR, k, PRES_VAR) ) / sRsL;
                    flY(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * U(i, jL, k, DENS_VAR) * U(i, jL, k, VELZ_VAR) &
                                                  - sL * vR * U(i, jR, k, DENS_VAR) * U(i, jR, k, VELZ_VAR) &
                                                  + sR*sL*(   U(i, jR, k, DENS_VAR) * U(i, jR, k, VELZ_VAR) &
                                                           -  U(i, jL, k, DENS_VAR) * U(i, jL, k, VELZ_VAR)) ) / sRsL
                    flY(i, j, k, HY_ENER_FLUX) = (  sR * vL * U(i, jL, k, DENS_VAR) * U(i, jL, k, ENER_VAR)  &
                                                  - sL * vR * U(i, jR, k, DENS_VAR) * U(i, jR, k, ENER_VAR)  &
                                                  + sR*sL*(   U(i, jR, k, DENS_VAR) * U(i, jR, k, ENER_VAR)  &
                                                           -  U(i, jL, k, DENS_VAR) * U(i, jL, k, ENER_VAR)) &
                                                  + sR * vL * U(i, jL, k, PRES_VAR)                          &
                                                  - sL * vR * U(i, jR, k, PRES_VAR) ) /sRsL
                end if

                flY(i, j, k, HY_DENS_FLUX) = flY(i, j, k, HY_DENS_FLUX) * dtdy
                flY(i, j, k, HY_XMOM_FLUX) = flY(i, j, k, HY_XMOM_FLUX) * dtdy
                flY(i, j, k, HY_YMOM_FLUX) = flY(i, j, k, HY_YMOM_FLUX) * dtdy
                flY(i, j, k, HY_ZMOM_FLUX) = flY(i, j, k, HY_ZMOM_FLUX) * dtdy
                flY(i, j, k, HY_ENER_FLUX) = flY(i, j, k, HY_ENER_FLUX) * dtdy
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_computeFluxesHll_Y_oacc

