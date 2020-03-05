program testOrchestrationRuntime
    use Orchestration_runtime_mod, only : Orchestration_init, &
                                          Orchestration_finalize, &
                                          Orchestration_executeTasks

    implicit none

    character(18), parameter :: LOG_FILENAME       = "TestRuntimeF90.log"
    integer,       parameter :: N_TEAMS            = 3
    integer,       parameter :: N_THREADS_PER_TEAM = 5

    call Orchestration_init(N_TEAMS, N_THREADS_PER_TEAM, LOG_FILENAME)
    call Orchestration_executeTasks(cpuTask)
    call Orchestration_finalize()

contains

    subroutine cpuTask(tId, work) bind(c)
        use iso_c_binding
        integer(c_int), intent(IN), value :: tId
        integer(c_int), intent(IN)        :: work

        write(*,"(A,I0,A,I0)") "[CPU Task/Thread ", tId, "] Work = ", work
    end subroutine cpuTask

end program testOrchestrationRuntime

