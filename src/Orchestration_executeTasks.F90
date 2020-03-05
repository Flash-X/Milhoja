!>
!!
!!
subroutine Orchestration_executeTasks(cpuTask)
    use iso_c_binding,           ONLY : c_funptr, c_funloc
    use Orchestration_interface, ONLY : Orchestration_runtimeTask
    use Orchestration_data,      ONLY : isRuntimeInitialized
    implicit none

    interface
        subroutine orchestration_execute_tasks_fi(cpuTask) bind(c)
            import
            implicit none
            type(c_funptr), intent(IN), value :: cpuTask
        end subroutine orchestration_execute_tasks_fi
    end interface

    procedure(Orchestration_runtimeTask) :: cpuTask

    type(c_funptr) :: cpuTaskPtr

    if (.NOT. isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has not been initialized" 
        STOP
    end if

    cpuTaskPtr = c_funloc(cpuTask)
    CALL orchestration_execute_tasks_fi(cpuTaskPtr)
end subroutine Orchestration_executeTasks

