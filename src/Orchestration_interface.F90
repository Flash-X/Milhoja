!>
!!
!! This is the header file for the Orchestation module that defines abstract
!! interfaces for use with procedure pointers as well as the public interface of
!! the unit.
!!

module Orchestration_interface
    implicit none
    public

    !!!!!----- DEFINE PROCEDURE POINTER INTERFACES
    abstract interface
        ! TODO: The unit of work is int.  After the AMReX iterator is built into
        !       the C++ code, we can swap this over to Grid_tile_t or a data
        !       packet of tiles
        subroutine Orchestration_runtimeTask(tId, work) bind(c)
            use iso_c_binding, ONLY : c_int
            integer(c_int), intent(IN), value :: tId
            integer(c_int), intent(IN)        :: work
        end subroutine Orchestration_runtimeTask
    end interface

    public :: Orchestration_runtimeTask

    !!!!!----- DEFINE GENERAL ROUTINE INTERFACES
    interface
        subroutine Orchestration_init(nTeams, nThreadsPerTeam, logFilename)
            integer,          intent(IN) :: nTeams
            integer,          intent(IN) :: nThreadsPerTeam
            character(len=*), intent(IN) :: logFilename 
        end subroutine Orchestration_init
    end interface

    interface
        subroutine Orchestration_finalize()
        end subroutine Orchestration_finalize
    end interface

    interface
        subroutine Orchestration_executeTasks(cpuTask)
            procedure(Orchestration_runtimeTask) :: cpuTask
        end subroutine Orchestration_executeTasks
    end interface

end module Orchestration_interface

