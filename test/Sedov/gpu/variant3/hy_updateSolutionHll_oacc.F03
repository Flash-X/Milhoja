#include "constants.h"
#include "Flash.h"

subroutine hy_updateSolutionHll_oacc(lo, hi, flX, flY, flZ, U)
    implicit none

    !$acc routine vector

    integer, intent(IN)    :: lo(:)
    integer, intent(IN)    :: hi(:)
    real,    intent(IN)    :: flX(:, :, :, :)
    real,    intent(IN)    :: flY(:, :, :, :)
    real,    intent(IN)    :: flZ(:, :, :, :)
    real,    intent(INOUT) :: U(:, :, :, :)

#ifdef EINT_VAR
    real :: norm2_sqr
#endif
    real :: densOld
    real :: densNew
    real :: densNew_inv

    integer :: i, j, k

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                ! Update density first
                densOld = U(i, j, k, DENS_VAR)
#if NDIM == 1
                densNew =   densOld                          &
                          + flX(i,   j, k, HY_DENS_FLUX)     &
                          - flX(i+1, j, k, HY_DENS_FLUX)
#elif NDIM == 2
                densNew =   densOld                          &
                          + flX(i,   j,   k, HY_DENS_FLUX)   &
                          - flX(i+1, j,   k, HY_DENS_FLUX)   &
                          + flY(i,   j,   k, HY_DENS_FLUX)   &
                          - flY(i,   j+1, k, HY_DENS_FLUX)
#elif NDIM == 3
                densNew =   densOld                          &
                          + flX(i,   j,   k,   HY_DENS_FLUX) &
                          - flX(i+1, j,   k,   HY_DENS_FLUX) &
                          + flY(i,   j,   k,   HY_DENS_FLUX) &
                          - flY(i,   j+1, k,   HY_DENS_FLUX) &
                          + flZ(i,   j,   k,   HY_DENS_FLUX) &
                          - flZ(i,   j,   k+1, HY_DENS_FLUX)
#endif
                U(i, j, k, DENS_VAR) = densNew
                densNew_inv = 1.0 / densNew

                ! velocities and total energy can be updated independently
                ! using density result
#if NDIM == 1
                U(i, j, k, VELX_VAR) = (    U(i,   j, k, VELX_VAR) * densOld     &
                                        + flX(i,   j, k, HY_XMOM_FLUX)           &
                                        - flX(i+1, j, k, HY_XMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELY_VAR) = (    U(i,   j, k, VELY_VAR) * densOld     &
                                        + flX(i,   j, k, HY_YMOM_FLUX)           &
                                        - flX(i+1, j, k, HY_YMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELZ_VAR) = (    U(i,   j, k, VELZ_VAR) * densOld     &
                                        + flX(i,   j, k, HY_ZMOM_FLUX)           &
                                        - flX(i+1, j, k, HY_ZMOM_FLUX) ) * densNew_inv

                U(i, j, k, ENER_VAR) = (    U(i,   j, k, ENER_VAR) * densOld     &
                                        + flX(i,   j, k, HY_ENER_FLUX)           &
                                        - flX(i+1, j, k, HY_ENER_FLUX) ) * densNew_inv
#elif NDIM == 2
                U(i, j, k, VELX_VAR) = (    U(i,   j,   k, VELX_VAR) * densOld   &
                                        + flX(i,   j,   k, HY_XMOM_FLUX)         &
                                        - flX(i+1, j,   k, HY_XMOM_FLUX)         &
                                        + flY(i,   j,   k, HY_XMOM_FLUX)         &
                                        - flY(i,   j+1, k, HY_XMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELY_VAR) = (    U(i,   j,   k, VELY_VAR) * densOld   &
                                        + flX(i,   j,   k, HY_YMOM_FLUX)         &
                                        - flX(i+1, j,   k, HY_YMOM_FLUX)         &
                                        + flY(i,   j,   k, HY_YMOM_FLUX)         &
                                        - flY(i,   j+1, k, HY_YMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELZ_VAR) = (    U(i,   j,   k, VELZ_VAR) * densOld   &
                                        + flX(i,   j,   k, HY_ZMOM_FLUX)         &
                                        - flX(i+1, j,   k, HY_ZMOM_FLUX)         &
                                        + flY(i,   j,   k, HY_ZMOM_FLUX)         &
                                        - flY(i,   j+1, k, HY_ZMOM_FLUX) ) * densNew_inv

                U(i, j, k, ENER_VAR) = (    U(i,   j,   k, ENER_VAR) * densOld   &
                                        + flX(i,   j,   k, HY_ENER_FLUX)         &
                                        - flX(i+1, j,   k, HY_ENER_FLUX)         &
                                        + flY(i,   j,   k, HY_ENER_FLUX)         &
                                        - flY(i,   j+1, k, HY_ENER_FLUX) ) * densNew_inv
#elif NDIM == 3
                U(i, j, k, VELX_VAR) = (    U(i,   j,   k,   VELX_VAR) * densOld &
                                        + flX(i,   j,   k,   HY_XMOM_FLUX)       &
                                        - flX(i+1, j,   k,   HY_XMOM_FLUX)       &
                                        + flY(i,   j,   k,   HY_XMOM_FLUX)       &
                                        - flY(i,   j+1, k,   HY_XMOM_FLUX)       &
                                        + flZ(i,   j,   k,   HY_XMOM_FLUX)       &
                                        - flZ(i,   j,   k+1, HY_XMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELY_VAR) = (    U(i,   j,   k,   VELY_VAR) * densOld &
                                        + flX(i,   j,   k,   HY_YMOM_FLUX)       &
                                        - flX(i+1, j,   k,   HY_YMOM_FLUX)       &
                                        + flY(i,   j,   k,   HY_YMOM_FLUX)       &
                                        - flY(i,   j+1, k,   HY_YMOM_FLUX)       &
                                        + flZ(i,   j,   k,   HY_YMOM_FLUX)       &
                                        - flZ(i,   j,   k+1, HY_YMOM_FLUX) ) * densNew_inv

                U(i, j, k, VELZ_VAR) = (    U(i,   j,   k,   VELZ_VAR) * densOld &
                                        + flX(i,   j,   k,   HY_ZMOM_FLUX)       &
                                        - flX(i+1, j,   k,   HY_ZMOM_FLUX)       &
                                        + flY(i,   j,   k,   HY_ZMOM_FLUX)       &
                                        - flY(i,   j+1, k,   HY_ZMOM_FLUX)       &
                                        + flZ(i,   j,   k,   HY_ZMOM_FLUX)       &
                                        - flZ(i,   j,   k+1, HY_ZMOM_FLUX) ) * densNew_inv

                U(i, j, k, ENER_VAR) = (    U(i,   j,   k,   ENER_VAR) * densOld &
                                        + flX(i,   j,   k,   HY_ENER_FLUX)       &
                                        - flX(i+1, j,   k,   HY_ENER_FLUX)       &
                                        + flY(i,   j,   k,   HY_ENER_FLUX)       &
                                        - flY(i,   j+1, k,   HY_ENER_FLUX)       &
                                        + flZ(i,   j,   k,   HY_ENER_FLUX)       &
                                        - flZ(i,   j,   k+1, HY_ENER_FLUX) ) * densNew_inv
#endif

#ifdef EINT_VAR
                ! Compute energy correction from new velocities and energy
                norm2_sqr =   U(i, j, k, VELX_VAR) * U(i, j, k, VELX_VAR) &
                            + U(i, j, k, VELY_VAR) * U(i, j, k, VELY_VAR) &
                            + U(i, j, k, VELZ_VAR) * U(i, j, k, VELZ_VAR)
                U(i, j, k, EINT_VAR) = U(i, j, k, ENER_VAR) - (0.5 * norm2_sqr);
#endif
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_updateSolutionHll_oacc

