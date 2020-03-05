program RuntimeTest_evolveFlash
    use Orchestration_interface, ONLY : Orchestration_init, &
                                        Orchestration_finalize, &
                                        Orchestration_executeTasks
    use RuntimeTest_data,        ONLY : task1_iteration, &
                                        task1_dt

    implicit none

    character(18), parameter :: LOG_FILENAME       = "TestRuntimeF90.log"
    integer,       parameter :: N_TEAMS            = 3
    integer,       parameter :: N_THREADS_PER_TEAM = 5

    call Orchestration_init(N_TEAMS, N_THREADS_PER_TEAM, LOG_FILENAME)

    ! Set task-specific data
    task1_iteration = 1
    task1_dt        = 1.0e-5
    call Orchestration_executeTasks(cpuTask)

    ! Set task-specific data
    task1_iteration = 2
    task1_dt        = 2.1e-5
    call Orchestration_executeTasks(cpuTask)

    call Orchestration_finalize()

contains

    subroutine cpuTask(tId, work) bind(c)
        use iso_c_binding
        use RuntimeTest_data, ONLY : task1_iteration, &
                                     task1_dt

        integer(c_int), intent(IN), value :: tId
        integer(c_int), intent(IN)        :: work

        write(*,"(A,I0,A,F15.7)")  "Iteration ", task1_iteration, " / dt = ", task1_dt
        write(*,"(A,I0,A,I0)")     "[CPU Task/Thread ", tId, "] Work = ", work
    end subroutine cpuTask

end program RuntimeTest_evolveFlash

