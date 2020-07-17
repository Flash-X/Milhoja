! DEV NOTE: this was developed as a way of testing the passing of
!           Fortran arrays to Grid.cpp. Not necessary in the long run.
! TODO: remove this file

#include "constants.h"
  
subroutine Grid_getCellVolumes(level, lo, hi, volumes)
    use iso_c_binding, ONLY : C_INT, C_DOUBLE

    use Grid_data,     ONLY : gr_isGridInitialized

    implicit none
    
    interface
        subroutine grid_get_cellvolumes_fi(lev,loVect,hiVect,vols) bind(c)
            import
            implicit none
            integer(C_INT), intent(IN) :: lev
            integer(C_INT), intent(IN) :: loVect(1:MDIM)
            integer(C_INT), intent(IN) :: hiVect(1:MDIM)
            real(C_DOUBLE), intent(out) :: vols(loVect(IAXIS):hiVect(IAXIS), &
                                                loVect(JAXIS):hiVect(JAXIS), &
                                                loVect(KAXIS):hiVect(KAXIS))
        end subroutine grid_get_cellvolumes_fi
    end interface

    integer, intent(in)  :: level
    integer, intent(in)  :: lo(1:MDIM)
    integer, intent(in)  :: hi(1:MDIM)
    real,    intent(out) :: volumes(lo(IAXIS):hi(IAXIS), &
                                    lo(JAXIS):hi(JAXIS), &
                                    lo(KAXIS):hi(KAXIS))

    real(C_DOUBLE) :: vols(lo(IAXIS):hi(IAXIS), &
                           lo(JAXIS):hi(JAXIS), &
                           lo(KAXIS):hi(KAXIS))
    integer        :: i, j, k

    if (.NOT. gr_isGridInitialized) then
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    ! AMReX uses 0-based level indexing; FLASH, 1-based.
    call grid_get_cellvolumes_fi(INT(level - 1, C_INT), &
                                 INT(lo , C_INT), &
                                 INT(hi , C_INT), &
                                 vols)


    volumes(:,:,:) = 0.0
    do i = lo(IAXIS),hi(IAXIS)
        do j = lo(JAXIS),hi(JAXIS)
            do k = lo(KAXIS),hi(KAXIS)
                volumes(i,j,k) = REAL(vols(i,j,k))
            end do
        end do
    end do

end subroutine Grid_getCellVolumes

