!>
!!
!! This is the header file for the Grid module that defines abstract
!! interfaces for use with procedure pointers as well as the public interface of
!! the unit.
!!

#include "constants.h"

module Grid_interface
    implicit none
    public

    !!!!!----- DEFINE PROCEDURE POINTER INTERFACES
    abstract interface
        subroutine setIcIface(tilePtr) bind(c)
            use iso_c_binding, ONLY : C_PTR
            type(C_PTR), intent(IN) :: tilePtr
        end subroutine setIcIface
    end interface

    !!!!!----- DEFINE GENERAL ROUTINE INTERFACES
    interface
        subroutine Grid_init()
        end subroutine Grid_init
    end interface

    interface
        subroutine Grid_initDomain(xMin, xMax, yMin, yMax, zMin, zMax, &
                                   nBlocksX, nBlocksY, nBlocksZ, &
                                   nVars)
            real,    intent(IN)   :: xMin
            real,    intent(IN)   :: xMax
            real,    intent(IN)   :: yMin
            real,    intent(IN)   :: yMax
            real,    intent(IN)   :: zMin
            real,    intent(IN)   :: zMax
            integer, intent(IN)   :: nBlocksX
            integer, intent(IN)   :: nBlocksY
            integer, intent(IN)   :: nBlocksZ
            integer, intent(IN)   :: nVars
        end subroutine Grid_initDomain
    end interface

    interface
        subroutine Grid_getDeltas(level, deltas)
            integer, intent(IN)  :: level
            real,    intent(OUT) :: deltas(1:MDIM)
        end subroutine Grid_getDeltas
    end interface

    interface
        subroutine Grid_getDomainBoundBox(bbox)
            real, intent(OUT) :: bbox(LOW:HIGH, 1:MDIM)
        end subroutine Grid_getDomainBoundBox
    end interface

    interface
        subroutine Grid_getCellCoords(axis, level, lo, hi, coordinates)
            integer, intent(IN)  :: axis
            integer, intent(IN)  :: level
            integer, intent(IN)  :: lo(1:MDIM)
            integer, intent(IN)  :: hi(1:MDIM)
            real,    intent(OUT) :: coordinates(:)
        end subroutine Grid_getCellCoords
    end interface

    interface
        subroutine Grid_finalize()
        end subroutine Grid_finalize
    end interface

end module Grid_interface

