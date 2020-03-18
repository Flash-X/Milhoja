#include "Flash.h"
#include "constants.h"

module tile_mod
    use iso_c_binding

    implicit none
    private

    type, public :: tile_t
        type(C_PTR) :: cptr         = C_NULL_PTR
        integer     :: gridIndex    = -1
        integer     :: lo(1:MDIM)   = 0
        integer     :: hi(1:MDIM)   = 0
        integer     :: loGC(1:MDIM) = 0
        integer     :: hiGC(1:MDIM) = 0
    contains
        generic            :: assignment(=) => shallow_copy_c_to_f
        procedure, public  :: dataPtr
        procedure, private :: shallow_copy_c_to_f
    end type

    interface
        subroutine tile_set_limits_fi(tilePtr, gid, lo, hi, loGC, hiGC) bind(c)
            use, intrinsic :: iso_c_binding
            type(C_PTR),    intent(IN), value :: tilePtr
            integer(C_INT), intent(OUT)       :: gid
            integer(C_INT), intent(OUT)       :: lo(1:MDIM)
            integer(C_INT), intent(OUT)       :: hi(1:MDIM)
            integer(C_INT), intent(OUT)       :: loGC(1:MDIM)
            integer(C_INT), intent(OUT)       :: hiGC(1:MDIM)
        end subroutine tile_set_limits_fi

        subroutine tile_get_data_ptr_fi(tilePtr, cptr) bind(c)
            import
            type(C_PTR), intent(IN), value :: tilePtr
            type(C_PTR), intent(OUT)       :: cptr
        end subroutine tile_get_data_ptr_fi
    end interface

contains

    subroutine shallow_copy_c_to_f(this, cptr)
        class(tile_t), intent(INOUT)       :: this
        type(C_PTR),   intent(IN),   value :: cptr

        integer(C_INT) :: gid
        integer(C_INT) :: lo(1:MDIM)
        integer(C_INT) :: hi(1:MDIM)
        integer(C_INT) :: loGC(1:MDIM)
        integer(C_INT) :: hiGC(1:MDIM)
        integer        :: i

        ! TODO: What to do if the pointer is already assigned in this?
        this%cptr = cptr

        call tile_set_limits_fi(this%cptr, gid, lo, hi, loGC, hiGC)

        this%gridIndex = INT(gid)

        this%lo(:)   = 1
        this%hi(:)   = 1
        this%loGC(:) = 1
        this%hiGC(:) = 1
        do i = 1, NDIM
            this%lo(i)   = INT(lo(i))
            this%hi(i)   = INT(hi(i))
            this%loGC(i) = INT(loGC(i))
            this%hiGC(i) = INT(hiGC(i))
        end do
    end subroutine shallow_copy_c_to_f

    subroutine dataPtr(this, ptr)
        use amrex_fort_module, ONLY : wp => amrex_real

        class(tile_t), intent(IN) :: this
        real, contiguous, pointer :: ptr(:, :, :, :)

        ! TODO: How to deal with conversion between real and amrex_real?
        type(C_PTR)                        :: cptr
        real(wp),      contiguous, pointer :: fptr(:, :, :, :)
        integer(C_INT)                     :: n(MDIM+1)

        n(1:MDIM) = this%hiGC(1:MDIM) - this%loGC(1:MDIM) + 1
        n(MDIM+1) = NUNKVAR

        call tile_get_data_ptr_fi(this%cptr, cptr)

        call C_F_POINTER(cptr, fptr, shape=n)
        ptr(this%loGC(IAXIS):, this%loGC(JAXIS):, this%loGC(KAXIS):, 1:) => fptr 
    end subroutine dataPtr

end module tile_mod

