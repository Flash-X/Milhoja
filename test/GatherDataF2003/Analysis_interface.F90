!>
!!
!! This is the header file for the Simulation module that defines abstract
!! interfaces for use with procedure pointers as well as the public interface of
!! the unit.
!!

module Analysis_interface
    implicit none
    public

    !!!!!----- DEFINE GENERAL ROUTINE INTERFACES
    interface
        subroutine Analysis_computeErrors(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Analysis_computeErrors
    end interface

end module Analysis_interface 

