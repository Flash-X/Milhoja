#include "Flash.h"
#include "constants.h"

subroutine Simulation_initBlock(tilePtr) bind(c)
    use iso_c_binding,  ONLY : C_PTR

    use Grid_interface, ONLY : Grid_getCellCoords
    use tile_mod,       ONLY : tile_t

    implicit none

    type(C_PTR), intent(IN), value :: tilePtr

    type(tile_t)                                :: tileDesc
    integer                                     :: loGC(1:MDIM)
    integer                                     :: hiGC(1:MDIM)
    real,                   contiguous, pointer :: f(:, :, :, :)
    real,       allocatable                     :: xCoords(:)
    real,       allocatable                     :: yCoords(:)
    real,       allocatable                     :: zCoords(:)

    real    :: x, y, z
    integer :: i, j, k

    tileDesc = tilePtr

    loGC = tileDesc%loGC
    hiGC = tileDesc%hiGC

    call tileDesc%dataPtr(f)

    allocate(xCoords(loGC(IAXIS):hiGC(IAXIS)), &
             yCoords(loGC(JAXIS):hiGC(JAXIS)), &
             zCoords(loGC(KAXIS):hiGC(KAXIS)))
    ! TODO: Get level from tileDesc
    call Grid_getCellCoords(IAXIS, 1, loGC, hiGC, xCoords)
    call Grid_getCellCoords(JAXIS, 1, loGC, hiGC, yCoords)
    call Grid_getCellCoords(KAXIS, 1, loGC, hiGC, zCoords)

    do         k = loGC(KAXIS), hiGC(KAXIS)
        z = zCoords(k)
        do     j = loGC(JAXIS), hiGC(JAXIS)
            y = yCoords(j)
            do i = loGC(IAXIS), hiGC(IAXIS)
                x = xCoords(i)
                f(i, j, k, DENS_VAR) =   3.0*x*x*x +     x*x + x &
                                       - 2.0*y*y*y - 1.5*y*y + y &
                                       + 5.0;
                f(i, j, k, ENER_VAR) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x &
                                       -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y &
                                       + 1.0;
            end do
        end do
    end do

    deallocate(xCoords, yCoords, zCoords)
end subroutine Simulation_initBlock

