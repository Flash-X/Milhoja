#include "Flash.h"
#include "constants.h"

subroutine Analysis_computeErrors(tId, tilePtr) bind(c)
    use iso_c_binding,  ONLY : C_INT, C_PTR

    use Grid_interface, ONLY : Grid_getCellCoords
    use tile_mod,       ONLY : tile_t
    use Analysis_data,  ONLY : an_LinfErrors, &
                               an_meanAbsErrors, &
                               an_energyFactor

    implicit none

    integer(C_INT), intent(IN), value :: tId
    type(C_PTR),    intent(IN), value :: tilePtr

    type(tile_t)                                :: tileDesc
    integer                                     :: idx
    integer                                     :: lo(1:MDIM)
    integer                                     :: hi(1:MDIM)
    real,                   contiguous, pointer :: f(:, :, :, :)
    real,       allocatable                     :: xCoords(:)
    real,       allocatable                     :: yCoords(:)
    real,       allocatable                     :: zCoords(:)

    real    :: fExpected
    real    :: absErr
    real    :: LinfDens
    real    :: LinfEner
    real    :: sumDens
    real    :: sumEner
    integer :: nCells

    real    :: x, y, z
    integer :: i, j, k

    nullify(f)

    tileDesc = tilePtr

    idx = tileDesc%gridIndex
    lo  = tileDesc%lo
    hi  = tileDesc%hi

    call tileDesc%dataPtr(f)

    allocate(xCoords(lo(IAXIS):hi(IAXIS)), &
             yCoords(lo(JAXIS):hi(JAXIS)), &
             zCoords(lo(KAXIS):hi(KAXIS)))
    ! TODO: Get level from tileDesc
    call Grid_getCellCoords(IAXIS, 1, lo, hi, xCoords)
    call Grid_getCellCoords(JAXIS, 1, lo, hi, yCoords)
    call Grid_getCellCoords(KAXIS, 1, lo, hi, zCoords)

    ! These variables are shared resources being accessed in parallel
    ! If we assume that the grid indices are unique and that each tile is
    ! only once, then there is no possibility for resource contention.
    nCells = 0
    sumDens = 0.0
    sumEner = 0.0
    LinfDens = 0.0
    LinfEner = 0.0
    do         k = lo(KAXIS), hi(KAXIS)
        z = zCoords(k)
        do     j = lo(JAXIS), hi(JAXIS)
            y = yCoords(j)
            do i = lo(IAXIS), hi(IAXIS)
                x = xCoords(i)

                fExpected =   3.0*x*x*x +     x*x + x &
                            - 2.0*y*y*y - 1.5*y*y + y &
                            + 5.0;
                absErr = ABS(f(i, j, k, DENS_VAR) - fExpected)
                sumDens = sumDens + absErr
                LinfDens = MAX(LinfDens, absErr)

                fExpected =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x &
                            -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y &
                            + 1.0;
                fExpected = fExpected * an_energyFactor
                absErr = ABS(f(i, j, k, ENER_VAR) - fExpected)
                sumEner = sumEner + absErr
                LinfEner = MAX(LinfEner, absErr)

                nCells = nCells + 1
            end do
        end do
    end do

    an_LinfErrors(DENS_VAR, idx) = LinfDens
    an_LinfErrors(ENER_VAR, idx) = LinfEner
    an_meanAbsErrors(DENS_VAR, idx) = sumDens / REAL(nCells)
    an_meanAbsErrors(ENER_VAR, idx) = sumEner / REAL(nCells)

    deallocate(xCoords, yCoords, zCoords)
    nullify(f)
end subroutine Analysis_computeErrors

