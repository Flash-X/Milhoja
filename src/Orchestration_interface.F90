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
        subroutine Orchestration_runtimeTask(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Orchestration_runtimeTask
    end interface

    public :: Orchestration_runtimeTask

    !!!!!----- DEFINE GENERAL ROUTINE INTERFACES
    interface
        subroutine Orchestration_init()
        end subroutine Orchestration_init
    end interface

    interface
        subroutine Orchestration_finalize()
        end subroutine Orchestration_finalize
    end interface

    interface
        subroutine Orchestration_executeTasks(cpuTask,     nCpuThreads, &
                                              gpuTask,     nGpuThreads, &
                                              postGpuTask, nPostGpuThreads)
            import :: Orchestration_runtimeTask
            procedure(Orchestration_runtimeTask), optional :: cpuTask
            integer, intent(IN),                  optional :: nCpuThreads
            procedure(Orchestration_runtimeTask), optional :: gpuTask
            integer, intent(IN),                  optional :: nGpuThreads
            procedure(Orchestration_runtimeTask), optional :: postGpuTask
            integer, intent(IN),                  optional :: nPostGpuThreads
        end subroutine Orchestration_executeTasks
    end interface

end module Orchestration_interface

