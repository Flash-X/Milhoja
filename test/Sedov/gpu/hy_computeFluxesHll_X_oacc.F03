#include "constants.h"
#include "Flash.h"

subroutine hy_computeFluxesHll_X_oacc(dt, lo, hi, deltas, U, auxC, flX)
    implicit none

    !$acc routine vector

    real,    intent(IN)  :: dt
    integer, intent(IN)  :: lo(:)
    integer, intent(IN)  :: hi(:)
    real,    intent(IN)  :: deltas(:)
    real,    intent(IN)  :: U(:, :, :, :)
    real,    intent(IN)  :: auxC(:, :, :)
    real,    intent(OUT) :: flX(:, :, :, :)

    real    :: sL
    real    :: sR
    real    :: sRsL
    real    :: vn
    real    :: vL
    real    :: vR
    integer :: is
    integer :: iL
    integer :: iR
    real    :: dtdx

    integer :: i, j, k

    dtdx = dt / deltas(IAXIS)

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)+K1D
                sL = MIN(U(i-1, j, k, VELX_VAR) - auxC(i-1, j, k), &
                         U(i,   j, k, VELX_VAR) - auxC(i,   j, k))
                sR = MAX(U(i-1, j, k, VELX_VAR) + auxC(i-1, j, k), &
                         U(i,   j, k, VELX_VAR) + auxC(i,   j, k))
                sRsL = sR - sL
                if (sL > 0.0) then
                    vn = U(i-1, j, k, VELX_VAR)
                    is = i - 1
                    iL = i - 1
                    iR = i - 1
                else if (sR < 0.0) then
                    vn = U(i, j, k, VELX_VAR)
                    is = i
                    iL = i
                    iR = i
                else
                    vn = 0.5 * (  U(i-1, j, k, VELX_VAR)  &
                                + U(i,   j, k, VELX_VAR))
                    is = i
                    iL = i-1
                    iR = i
                    if (vn > 0.0) then
                        is = is - 1
                    end if 
                end if

                vL = U(iL, j, k, VELX_VAR)
                vR = U(iR, j, k, VELX_VAR)
                if (iL == iR) then
                    flX(i, j, k, HY_DENS_FLUX) =   vn * U(is, j, k, DENS_VAR)
                    flX(i, j, k, HY_XMOM_FLUX) =   vn * U(is, j, k, DENS_VAR) &
                                                      * U(is, j, k, VELX_VAR) &
                                                 +      U(is, j, k, PRES_VAR)
                    flX(i, j, k, HY_YMOM_FLUX) =   vn * U(is, j, k, DENS_VAR) &
                                                      * U(is, j, k, VELY_VAR)
                    flX(i, j, k, HY_ZMOM_FLUX) =   vn * U(is, j, k, DENS_VAR) &
                                                      * U(is, j, k, VELZ_VAR)
                    flX(i, j, k, HY_ENER_FLUX) =   vn * U(is, j, k, DENS_VAR) &
                                                      * U(is, j, k, ENER_VAR) &
                                                 + vn * U(is, j, k, PRES_VAR)
                else
                    flX(i, j, k, HY_DENS_FLUX) = (  sR * vL * U(iL, j, k, DENS_VAR) &
                                                  - sL * vR * U(iR, j, k, DENS_VAR) &
                                                  + sR*sL*(   U(iR, j, k, DENS_VAR) &
                                                            - U(iL, j, k, DENS_VAR)) ) / sRsL
                    flX(i, j, k, HY_XMOM_FLUX) = (  sR * vL * U(iL, j, k, DENS_VAR) * U(iL, j, k, VELX_VAR)  &
                                                  - sL * vR * U(iR, j, k, DENS_VAR) * U(iR, j, k, VELX_VAR)  &
                                                  + sR*sL*(   U(iR, j, k, DENS_VAR) * U(iR, j, k, VELX_VAR)  &
                                                            - U(iL, j, k, DENS_VAR) * U(iL, j, k, VELX_VAR)) &
                                                  + sR *      U(iL, j, k, PRES_VAR)                          &
                                                  - sL *      U(iR, j, k, PRES_VAR) ) / sRsL
                    flX(i, j, k, HY_YMOM_FLUX) = (  sR * vL * U(iL, j, k, DENS_VAR) * U(iL, j, k, VELY_VAR) &
                                                  - sL * vR * U(iR, j, k, DENS_VAR) * U(iR, j, k, VELY_VAR) &
                                                  + sR*sL*(   U(iR, j, k, DENS_VAR) * U(iR, j, k, VELY_VAR) &
                                                            - U(iL, j, k, DENS_VAR) * U(iL, j, k, VELY_VAR)) ) / sRsL
                    flX(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * U(iL, j, k, DENS_VAR) * U(iL, j, k, VELZ_VAR) &
                                                  - sL * vR * U(iR, j, k, DENS_VAR) * U(iR, j, k, VELZ_VAR) &
                                                  + sR*sL*(   U(iR, j, k, DENS_VAR) * U(iR, j, k, VELZ_VAR) &
                                                            - U(iL, j, k, DENS_VAR) * U(iL, j, k, VELZ_VAR)) ) / sRsL
                    flX(i, j, k, HY_ENER_FLUX) = (  sR * vL * U(iL, j, k, DENS_VAR) * U(iL, j, k, ENER_VAR)  &
                                                  - sL * vR * U(iR, j, k, DENS_VAR) * U(iR, j, k, ENER_VAR)  &
                                                  + sR*sL*(   U(iR, j, k, DENS_VAR) * U(iR, j, k, ENER_VAR)  &
                                                            - U(iL, j, k, DENS_VAR) * U(iL, j, k, ENER_VAR)) &
                                                  + sR * vL * U(iL, j, k, PRES_VAR)                          &
                                                  - sL * vR * U(iR, j, k, PRES_VAR) ) / sRsL
                end if

                flX(i, j, k, HY_DENS_FLUX) = flX(i, j, k, HY_DENS_FLUX) * dtdx
                flX(i, j, k, HY_XMOM_FLUX) = flX(i, j, k, HY_XMOM_FLUX) * dtdx
                flX(i, j, k, HY_YMOM_FLUX) = flX(i, j, k, HY_YMOM_FLUX) * dtdx
                flX(i, j, k, HY_ZMOM_FLUX) = flX(i, j, k, HY_ZMOM_FLUX) * dtdx
                flX(i, j, k, HY_ENER_FLUX) = flX(i, j, k, HY_ENER_FLUX) * dtdx
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_computeFluxesHll_X_oacc

