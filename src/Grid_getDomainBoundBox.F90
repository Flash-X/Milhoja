#include "constants.h"

subroutine Grid_getDomainBoundBox(bbox)
    use iso_c_binding, ONLY : C_DOUBLE

    use Grid_data,     ONLY : gr_isGridInitialized

    implicit none

    interface
        subroutine grid_get_domain_bound_box_fi(lo, hi) bind(c)
            use iso_c_binding, ONLY : C_DOUBLE
            implicit none
            real(C_DOUBLE), intent(OUT) :: lo(1:NDIM)
            real(C_DOUBLE), intent(OUT) :: hi(1:NDIM)
        end subroutine grid_get_domain_bound_box_fi
    end interface

    real, intent(OUT) :: bbox(LOW:HIGH, 1:MDIM)

    real(C_DOUBLE) :: lo(1:NDIM)
    real(C_DOUBLE) :: hi(1:NDIM)
    integer        :: i

    if (.NOT. gr_isGridInitialized) then
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    call grid_get_domain_bound_box_fi(lo, hi)
    bbox(:, :) = 0.0
    do i = 1, NDIM
        bbox(LOW,  i) = REAL(lo(i))
        bbox(HIGH, i) = REAL(hi(i))
    end do
end subroutine Grid_getDomainBoundBox

