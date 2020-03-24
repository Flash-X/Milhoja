!>
!!
!!
subroutine Orchestration_executeTasks(cpuTask,     nCpuThreads, &
                                      gpuTask,     nGpuThreads, &
                                      postGpuTask, nPostGpuThreads)
    use iso_c_binding,           ONLY : C_INT, C_FUNPTR, C_FUNLOC, C_NULL_FUNPTR

    use Orchestration_interface, ONLY : Orchestration_runtimeTask
    use Orchestration_data,      ONLY : or_isRuntimeInitialized

    implicit none

    interface
        function orchestration_execute_tasks_fi(cpuTask,     nCpuThreads, &
                                                gpuTask,     nGpuThreads, &
                                                postGpuTask, nPostGpuThreads) &
                                                result(success) bind(c)
            import
            implicit none
            type(C_FUNPTR), intent(IN), value :: cpuTask
            integer(C_INT), intent(IN), value :: nCpuThreads
            type(C_FUNPTR), intent(IN), value :: gpuTask
            integer(C_INT), intent(IN), value :: nGpuThreads
            type(C_FUNPTR), intent(IN), value :: postGpuTask
            integer(C_INT), intent(IN), value :: nPostGpuThreads
            integer(C_INT)                    :: success
        end function orchestration_execute_tasks_fi
    end interface

    procedure(Orchestration_runtimeTask), optional :: cpuTask
    integer, intent(IN),                  optional :: nCpuThreads
    procedure(Orchestration_runtimeTask), optional :: gpuTask
    integer, intent(IN),                  optional :: nGpuThreads
    procedure(Orchestration_runtimeTask), optional :: postGpuTask
    integer, intent(IN),                  optional :: nPostGpuThreads

    type(C_FUNPTR) :: cpuTask_C
    type(C_FUNPTR) :: gpuTask_C
    type(C_FUNPTR) :: postGpuTask_C

    integer(C_INT) :: nCpuThreads_C
    integer(C_INT) :: nGpuThreads_C
    integer(C_INT) :: nPostGpuThreads_C

    integer(C_INT) :: success

    if (.NOT. or_isRuntimeInitialized) then
        write(*,*) "The Orchestration Runtime has not been initialized" 
        STOP
    end if

    cpuTask_C = C_NULL_FUNPTR
    nCpuThreads_C = 0
    if (present(cpuTask)) then
        cpuTask_C = C_FUNLOC(cpuTask)
        if (.NOT. present(nCpuThreads)) then
            write(*,*) "[Orchestration_executeTasks] Missing nCpuThreads"
            STOP
        end if
        nCpuThreads_C = INT(nCpuThreads, C_INT)
    end if

    gpuTask_C = C_NULL_FUNPTR
    nGpuThreads_C = 0
    if (present(gpuTask)) then
        gpuTask_C = C_FUNLOC(gpuTask)
        if (.NOT. present(nGpuThreads)) then
            write(*,*) "[Orchestration_executeTasks] Missing nGpuThreads"
            STOP
        end if
        nGpuThreads_C = INT(nGpuThreads, C_INT)
    end if

    postGpuTask_C = C_NULL_FUNPTR
    nPostGpuThreads_C = 0
    if (present(postGpuTask)) then
        postGpuTask_C = C_FUNLOC(postGpuTask)
        if (.NOT. present(nPostGpuThreads)) then
            write(*,*) "[Orchestration_executeTasks] Missing nPostGpuThreads"
            STOP
        end if
        nPostGpuThreads_C = INT(nPostGpuThreads, C_INT)
    end if

    success = orchestration_execute_tasks_fi(cpuTask_C,     nCpuThreads_C, &
                                             gpuTask_C,     nGpuThreads_C, &
                                             postGpuTask_C, nPostGpuThreads_C)
    if (success /= 1) then
        write(*,*) "[Orchestration_executeTasks] Unable to execute tasks"
        STOP
    end if
end subroutine Orchestration_executeTasks

