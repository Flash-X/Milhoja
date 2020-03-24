#include "Flash.h"
#include "constants.h"

subroutine Physics_op1_executeTask3_Tile(tId, tilePtr) bind(c)
    use iso_c_binding, ONLY : C_INT, C_PTR

    use Physics_data,  ONLY : ph_op1_energyFactor
    use tile_mod,      ONLY : tile_t

    implicit none

    integer(C_INT), intent(IN), value :: tId
    type(C_PTR),    intent(IN), value :: tilePtr

    type(tile_t)         :: tileDesc
    real,        pointer :: f(:, :, :, :)
    integer              :: lo(1:MDIM) 
    integer              :: hi(1:MDIM)
    integer              :: i, j, k

    nullify(f)

    tileDesc = tilePtr

    call tileDesc%dataPtr(f)
    lo = tileDesc%lo(:)
    hi = tileDesc%hi(:)
    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                f(i, j, k, ENER_VAR) = ph_op1_energyFactor * f(i, j, k, ENER_VAR)
            end do
        end do
    end do
    nullify(f)
end subroutine Physics_op1_executeTask3_Tile

