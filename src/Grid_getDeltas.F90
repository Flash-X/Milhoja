#include "constants.h"
  
subroutine Grid_getDeltas(level, deltas)
    use iso_c_binding, ONLY : C_INT, C_DOUBLE

    use Grid_data,     ONLY : gr_isGridInitialized

    implicit none
    
    interface
        subroutine grid_get_deltas_fi(level, deltas) bind(c)
            import
            implicit none
            integer(C_INT), intent(IN), value :: level
            real(C_DOUBLE), intent(OUT)       :: deltas(1:NDIM)
        end subroutine grid_get_deltas_fi
    end interface

    integer, intent(IN)  :: level
    real,    intent(OUT) :: deltas(1:MDIM)

    real(C_DOUBLE) :: deltasC(1:NDIM)
    integer        :: i

    if (.NOT. gr_isGridInitialized) then
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    ! AMReX uses 0-based level indexing; FLASH, 1-based.
    call grid_get_deltas_fi(int(level - 1, C_INT), deltasC)

    ! TODO: Try to avoid the double looping that presently happens
    deltas(:) = 0.0
    do i = 1, NDIM
        deltas(i) = REAL(deltasC(i))
    end do
end subroutine Grid_getDeltas

