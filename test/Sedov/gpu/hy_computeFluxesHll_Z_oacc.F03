#include "constants.h"
#include "Flash.h"

subroutine hy_computeFluxesHll_Z_oacc(dt, lo, hi, deltas, U, auxC, flZ)
    implicit none

    !$acc routine vector

    real,    intent(IN)  :: dt
    integer, intent(IN)  :: lo(:)
    integer, intent(IN)  :: hi(:)
    real,    intent(IN)  :: deltas(:)
    real,    intent(IN)  :: U(:, :, :, :)
    real,    intent(IN)  :: auxC(:, :, :)
    real,    intent(OUT) :: flZ(:, :, :, :)

    real    :: sL
    real    :: sR
    real    :: sRsL
    real    :: vn
    real    :: vL
    real    :: vR
    integer :: ks
    integer :: kL
    integer :: kR
    real    :: dtdz

    integer :: i, j, k

    dtdz = dt / deltas(KAXIS)

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS), hi(KAXIS)+K3D
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                sL = MIN(U(i, j, k-1, VELZ_VAR) - auxC(i, j, k-1), &
                         U(i, j, k,   VELZ_VAR) - auxC(i, j, k))
                sR = MAX(U(i, j, k-1, VELZ_VAR) + auxC(i, j, k-1), &
                         U(i, j, k,   VELZ_VAR) + auxC(i, j, k))
                sRsL = sR - sL
                if (sL > 0.0) then
                    vn = U(i, j, k-1, VELZ_VAR)
                    ks = k - 1
                    kL = k - 1
                    kR = k - 1
                else if (sR < 0.0) then
                    vn = U(i, j, k, VELZ_VAR)
                    ks = k
                    kL = k
                    kR = k
                else
                    vn = 0.5 * (  U(i, j, k-1, VELZ_VAR)  &
                                + U(i, j, k,   VELZ_VAR))
                    ks = k
                    kL = k-1
                    kR = k
                    if (vn > 0.0) then
                      ks = ks - 1
                    end if
                end if

                vL = U(i, j, kL, VELZ_VAR)
                vR = U(i, j, kR, VELZ_VAR)
                if (kL == kR) then
                    flZ(i, j, k, HY_DENS_FLUX) =   vn * U(i, j, ks, DENS_VAR)
                    flZ(i, j, k, HY_XMOM_FLUX) =   vn * U(i, j, ks, DENS_VAR) &
                                                      * U(i, j, ks, VELX_VAR)
                    flZ(i, j, k, HY_YMOM_FLUX) =   vn * U(i, j, ks, DENS_VAR) &
                                                      * U(i, j, ks, VELY_VAR)
                    flZ(i, j, k, HY_ZMOM_FLUX) =   vn * U(i, j, ks, DENS_VAR) &
                                                      * U(i, j, ks, VELZ_VAR) &
                                                 +      U(i, j, ks, PRES_VAR)
                    flZ(i, j, k, HY_ENER_FLUX) =   vn * U(i, j, ks, DENS_VAR) &
                                                      * U(i, j, ks, ENER_VAR) &
                                                 + vn * U(i, j, ks, PRES_VAR)
                else
                    flZ(i, j, k, HY_DENS_FLUX) = (  sR * vL * U(i, j, kL, DENS_VAR) &
                                                  - sL * vR * U(i, j, kR, DENS_VAR) &
                                                  + sR*sL*(   U(i, j, kR, DENS_VAR) &
                                                           -  U(i, j, kL, DENS_VAR))) / sRsL
                    flZ(i, j, k, HY_XMOM_FLUX) = (  sR * vL * U(i, j, kL, DENS_VAR) * U(i, j, kL, VELX_VAR) &
                                                  - sL * vR * U(i, j, kR, DENS_VAR) * U(i, j, kR, VELX_VAR) &
                                                  + sR*sL*(   U(i, j, kR, DENS_VAR) * U(i, j, kR, VELX_VAR) &
                                                           -  U(i, j, kL, DENS_VAR) * U(i, j, kL, VELX_VAR)) ) / sRsL
                    flZ(i, j, k, HY_YMOM_FLUX) = (  sR * vL * U(i, j, kL, DENS_VAR) * U(i, j, kL, VELY_VAR) &
                                                  - sL * vR * U(i, j, kR, DENS_VAR) * U(i, j, kR, VELY_VAR) &
                                                  + sR*sL*(   U(i, j, kR, DENS_VAR) * U(i, j, kR, VELY_VAR) &
                                                           -  U(i, j, kL, DENS_VAR) * U(i, j, kL, VELY_VAR)) ) / sRsL
                    flZ(i, j, k, HY_ZMOM_FLUX) = (  sR * vL * U(i, j, kL, DENS_VAR) * U(i, j, kL, VELZ_VAR)  &
                                                  - sL * vR * U(i, j, kR, DENS_VAR) * U(i, j, kR, VELZ_VAR)  &
                                                  + sR*sL*(   U(i, j, kR, DENS_VAR) * U(i, j, kR, VELZ_VAR)  &
                                                           -  U(i, j, kL, DENS_VAR) * U(i, j, kL, VELZ_VAR)) &
                                                  + sR *      U(i, j, kL, PRES_VAR)                          &
                                                  - sL *      U(i, j, kR, PRES_VAR) ) / sRsL
                    flZ(i, j, k, HY_ENER_FLUX) = (  sR * vL * U(i, j, kL, DENS_VAR) * U(i, j, kL, ENER_VAR)  &
                                                  - sL * vR * U(i, j, kR, DENS_VAR) * U(i, j, kR, ENER_VAR)  &
                                                  + sR*sL*(   U(i, j, kR, DENS_VAR) * U(i, j, kR, ENER_VAR)  &
                                                           -  U(i, j, kL, DENS_VAR) * U(i, j, kL, ENER_VAR)) &
                                                  + sR * vL * U(i, j, kL, PRES_VAR)                          &
                                                  - sL * vR * U(i, j, kR, PRES_VAR) ) / sRsL
                end if

                flZ(i, j, k, HY_DENS_FLUX) = flZ(i, j, k, HY_DENS_FLUX) * dtdz
                flZ(i, j, k, HY_XMOM_FLUX) = flZ(i, j, k, HY_XMOM_FLUX) * dtdz
                flZ(i, j, k, HY_YMOM_FLUX) = flZ(i, j, k, HY_YMOM_FLUX) * dtdz
                flZ(i, j, k, HY_ZMOM_FLUX) = flZ(i, j, k, HY_ZMOM_FLUX) * dtdz
                flZ(i, j, k, HY_ENER_FLUX) = flZ(i, j, k, HY_ENER_FLUX) * dtdz
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_computeFluxesHll_Z_oacc

