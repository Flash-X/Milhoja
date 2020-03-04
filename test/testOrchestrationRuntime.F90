program testOrchestrationRuntime
    use Orchestration_runtime_mod, only : Orchestration_init, &
                                          Orchestration_finalize

    implicit none

    character(18), parameter :: LOG_FILENAME       = "TestRuntimeF90.log"
    integer,       parameter :: N_TEAMS            = 3
    integer,       parameter :: N_THREADS_PER_TEAM = 5

    call Orchestration_init(N_TEAMS, N_THREADS_PER_TEAM, LOG_FILENAME)
    call Orchestration_finalize()

end program testOrchestrationRuntime

