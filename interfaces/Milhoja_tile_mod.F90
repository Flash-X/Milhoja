!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level Fortan interface for interacting with tiles.
module milhoja_tile_mod
    implicit none
    private

    !!!!!----- PUBLIC INTERFACE
    public :: milhoja_tile_from_wrapper_C

    !!!!!----- INTERFACES TO C-LINKAGE C++ FUNCTIONS
    ! The C-to-Fortran interoperability layer
    interface
        !> Fortran interface on routine in C interface of same name.
        function milhoja_tile_from_wrapper_C(C_dataItemPtr, C_tilePtr) &
                                             result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_dataItemPtr
            type(C_PTR),          intent(OUT)       :: C_tilePtr
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_tile_from_wrapper_C
    end interface

end module milhoja_tile_mod

