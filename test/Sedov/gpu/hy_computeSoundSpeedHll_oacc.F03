#include "constants.h"
#include "Flash.h"

subroutine hy_computeSoundSpeedHll_oacc(lo, hi, U, auxC)
    implicit none

    !$acc routine vector

    integer, intent(IN)  :: lo(:)
    integer, intent(IN)  :: hi(:)
    real,    intent(IN)  :: U(:, :, :, :)
    real,    intent(OUT) :: auxC(:, :, :)

    integer :: i, j, k

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS)-K3D, hi(KAXIS)+K3D
        do     j = lo(JAXIS)-K2D, hi(JAXIS)+K2D
            do i = lo(IAXIS)-K1D, hi(IAXIS)+K1D
                auxC(i, j, k) = SQRT(  U(i, j, k, GAMC_VAR)   &
                                     * U(i, j, k, PRES_VAR)   &
                                     / U(i, j, k, DENS_VAR) )
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_computeSoundSpeedHll_oacc

