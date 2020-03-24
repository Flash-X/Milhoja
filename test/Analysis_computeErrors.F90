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

    type(tile_t)                    :: tileDesc
    integer                         :: idx
    integer                         :: level
    integer                         :: lo(1:MDIM)
    integer                         :: hi(1:MDIM)
    real,                   pointer :: f(:, :, :, :)
    real,       allocatable         :: xCoords(:)
    real,       allocatable         :: yCoords(:)
    real,       allocatable         :: zCoords(:)

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

    idx   = tileDesc%gridIndex
    level = tileDesc%level
    lo    = tileDesc%lo
    hi    = tileDesc%hi

    call tileDesc%dataPtr(f)

    allocate(xCoords(lo(IAXIS):hi(IAXIS)), &
             yCoords(lo(JAXIS):hi(JAXIS)), &
             zCoords(lo(KAXIS):hi(KAXIS)))
    call Grid_getCellCoords(IAXIS, level, lo, hi, xCoords)
    call Grid_getCellCoords(JAXIS, level, lo, hi, yCoords)
    call Grid_getCellCoords(KAXIS, level, lo, hi, zCoords)

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

                fExpected = 18.0*x - 12.0*y - 1.0
                absErr = ABS(f(i, j, k, DENS_VAR) - fExpected)
                sumDens = sumDens + absErr
                LinfDens = MAX(LinfDens, absErr)

                fExpected = an_energyFactor * (  48.0*x*x - 18.0*x &
                                               - 12.0*y*y + 12.0*y &
                                               - 2.0)
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

