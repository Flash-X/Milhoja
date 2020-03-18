!>
!!
!!
subroutine Orchestration_executeTasks(cpuTask, nCpuThreads)
    use iso_c_binding,           ONLY : C_INT, C_FUNPTR, C_FUNLOC

    use Orchestration_interface, ONLY : Orchestration_runtimeTask
    use Orchestration_data,      ONLY : or_isRuntimeInitialized

    implicit none

    interface
        subroutine orchestration_execute_tasks_fi(cpuTask, nCpuTasks) bind(c)
            import
            implicit none
            type(C_FUNPTR), intent(IN), value :: cpuTask
            integer(C_INT), intent(IN), value :: nCpuTasks
        end subroutine orchestration_execute_tasks_fi
    end interface

    procedure(Orchestration_runtimeTask) :: cpuTask
    integer, intent(IN)                  :: nCpuThreads

    if (.NOT. or_isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has not been initialized" 
        STOP
    end if

    CALL orchestration_execute_tasks_fi(C_FUNLOC(cpuTask), &
                                        INT(nCpuThreads, C_INT))
end subroutine Orchestration_executeTasks

