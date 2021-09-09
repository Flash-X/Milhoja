#include "constants.h"

subroutine hy_kernel_3_packet_oacc(lo, hi, U)
    use openacc

    implicit none

    !$acc routine vector

    integer, intent(IN)    :: lo(:)
    integer, intent(IN)    :: hi(:)
    real,    intent(INOUT) :: U(:, :, :, :)

    integer :: i, j, k, var

    !$acc loop vector collapse(4)
    do           var = 1, 8
        do         k = lo(KAXIS), hi(KAXIS)
            do     j = lo(JAXIS), hi(JAXIS)
                do i = lo(IAXIS), hi(IAXIS)
                    U(i, j, k, var) = U(i, j, k, var) * var
                end do
            end do
        end do
    end do
    !$acc end loop

end subroutine hy_kernel_3_packet_oacc

