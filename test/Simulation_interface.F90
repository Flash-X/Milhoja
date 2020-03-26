!>
!!
!! This is the header file for the Simulation module that defines abstract
!! interfaces for use with procedure pointers as well as the public interface of
!! the unit.
!!

module Simulation_interface
    implicit none
    public

    !!!!!----- DEFINE GENERAL ROUTINE INTERFACES
    interface
        subroutine Simulation_initBlock(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Simulation_initBlock
    end interface

end module Simulation_interface

