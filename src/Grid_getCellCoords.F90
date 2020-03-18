#include "constants.h"

subroutine Grid_getCellCoords(axis, level, lo, hi, coordinates)
    use Grid_interface,   ONLY : Grid_getDeltas
    use Grid_data,        ONLY : gr_globalDomain

    implicit none

    integer, intent(IN)  :: axis
    integer, intent(IN)  :: level
    integer, intent(IN)  :: lo(1:MDIM)
    integer, intent(IN)  :: hi(1:MDIM)
    real,    intent(OUT) :: coordinates(:)

    real    :: x0
    real    :: dx
    real    :: shift
    integer :: nElements
    real    :: deltas(1:MDIM)
    integer :: i

    nElements = hi(axis) - lo(axis) + 1
    if (SIZE(coordinates) < nElements) then
        write(*,*) "[Grid_getCellCoords] coordinates is too small"
        STOP
    end if

    call Grid_getDeltas(level, deltas)

    shift = 1.5
    x0 = gr_globalDomain(LOW, axis)
    dx = deltas(axis)
    do i = 1, nElements
        coordinates(i) = x0 + (lo(axis) + i - shift) * dx
    end do
end subroutine Grid_getCellCoords

