#include "constants.h"
#include "Flash.h"

subroutine eos_idealGammaDensIe_oacc(lo, hi, U)
    use Eos_data, ONLY : Eos_gamma

    implicit none

    !$acc routine vector

    ! Taken from Flash-X Sedov/setup_params file 
    real, parameter :: SINGLE_SPECIES_A = 1.0

    ! Taken from Flash-X Physical Constants
    real, parameter :: GAS_CONSTANT = 8.3144598e7  ! J/mol/K

    integer, intent(IN)    :: lo(:)
    integer, intent(IN)    :: hi(:)
    real,    intent(INOUT) :: U(:, :, :, :)

    real    :: ggprod_inv
    real    :: gamma_m_1_inv
    integer :: i, j, k

    ggprod_inv    = (Eos_gamma - 1.0) / GAS_CONSTANT
    gamma_m_1_inv =  Eos_gamma - 1.0

    !$acc loop vector collapse(3)
    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                U(i, j, k, PRES_VAR) =   U(i, j, k, DENS_VAR) &
                                       * U(i, j, k, EINT_VAR) &
                                       * gamma_m_1_inv
                U(i, j, k, TEMP_VAR) =   U(i, j, k, EINT_VAR) &
                                       * ggprod_inv           &
                                       * SINGLE_SPECIES_A
            end do
        end do
    end do
    !$acc end loop

end subroutine eos_idealGammaDensIe_oacc

