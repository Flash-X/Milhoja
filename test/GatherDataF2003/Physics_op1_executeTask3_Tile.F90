#include "Flash.h"
#include "constants.h"

subroutine Physics_op1_executeTask3_Tile(tId, tilePtr) bind(c)
    use iso_c_binding, ONLY : C_INT, C_PTR

    use Grid_interface, ONLY : Grid_getCellCoords
    use Physics_data,   ONLY : ph_op1_energyFactor
    use tile_mod,       ONLY : tile_t

    implicit none

    integer(C_INT), intent(IN), value :: tId
    type(C_PTR),    intent(IN), value :: tilePtr

    type(tile_t)                    :: tileDesc
    integer                         :: level
    integer                         :: lo(1:MDIM) 
    integer                         :: hi(1:MDIM)
    real,                   pointer :: f(:, :, :, :)
    real,       allocatable         :: xCoords(:)
    real,       allocatable         :: yCoords(:)

    real    :: x, y, z
    integer :: i, j, k

    nullify(f)

    tileDesc = tilePtr

    lo    = tileDesc%lo(:)
    hi    = tileDesc%hi(:)
    level = tileDesc%level

    call tileDesc%dataPtr(f)

    allocate(xCoords(lo(IAXIS):hi(IAXIS)), &
             yCoords(lo(JAXIS):hi(JAXIS)))
    call Grid_getCellCoords(IAXIS, level, lo, hi, xCoords)
    call Grid_getCellCoords(JAXIS, level, lo, hi, yCoords)

    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            y = yCoords(j)
            do i = lo(IAXIS), hi(IAXIS)
                x = xCoords(i)
                f(i, j, k, ENER_VAR) =   ph_op1_energyFactor * x * y &
                                       * f(i, j, k, ENER_VAR)
            end do
        end do
    end do
    nullify(f)

    deallocate(xCoords, yCoords)
end subroutine Physics_op1_executeTask3_Tile

