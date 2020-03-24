#include "Flash.h"
#include "constants.h"

subroutine Physics_op1_executeTask2_Tile(tId, tilePtr) bind(c)
    use iso_c_binding,  ONLY : C_INT, C_PTR
    
    use Grid_interface, ONLY : Grid_getDeltas
    use tile_mod,       ONLY : tile_t

    integer(C_INT), intent(IN), value :: tId
    type(C_PTR),    intent(IN), value :: tilePtr

    type(tile_t)                     :: tileDesc
    real,        contiguous, pointer :: Uin(:, :, :, :)
    integer                          :: lo(1:MDIM) 
    integer                          :: hi(1:MDIM)
    real                             :: deltas(1:MDIM)
    real                             :: dx_sqr_inv
    real                             :: dy_sqr_inv
    integer                          :: i, j, k

    real, allocatable :: Uout(:, :, :)

    real :: u_ij
    real :: u_im1_j
    real :: u_ip1_j
    real :: u_i_jm1
    real :: u_i_jp1

    nullify(Uin)

    tileDesc = tilePtr

    call tileDesc%dataPtr(Uin)
    lo = tileDesc%lo(:)
    hi = tileDesc%hi(:)
    call Grid_getDeltas(tileDesc%level, deltas)

    dx_sqr_inv = 1.0 / (deltas(IAXIS) * deltas(IAXIS))
    dy_sqr_inv = 1.0 / (deltas(JAXIS) * deltas(JAXIS))

    allocate(Uout(lo(IAXIS):hi(IAXIS), &
                  lo(JAXIS):hi(JAXIS), &
                  lo(KAXIS):hi(KAXIS)))

    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                u_ij    = Uin(i  , j  , k, ENER_VAR)
                u_im1_j = Uin(i-1, j  , k, ENER_VAR)
                u_ip1_j = Uin(i+1, j  , k, ENER_VAR)
                u_i_jm1 = Uin(i  , j-1, k, ENER_VAR)
                u_i_jp1 = Uin(i  , j+1, k, ENER_VAR)
                Uout(i, j, k) =   ((u_im1_j + u_ip1_j) - 2.0*u_ij) * dx_sqr_inv &
                                + ((u_i_jm1 + u_i_jp1) - 2.0*u_ij) * dy_sqr_inv
            end do
        end do
    end do

    do         k = lo(KAXIS), hi(KAXIS)
        do     j = lo(JAXIS), hi(JAXIS)
            do i = lo(IAXIS), hi(IAXIS)
                Uin(i, j, k, ENER_VAR) = Uout(i, j, k)
            end do
        end do
    end do
    nullify(Uin)
    deallocate(Uout)
end subroutine Physics_op1_executeTask2_Tile

