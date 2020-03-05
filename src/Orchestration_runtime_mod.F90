module Orchestration_runtime_mod 
    use iso_c_binding

    implicit none
    private

    logical, save :: isInitialized = .FALSE.

    public :: Orchestration_init
    public :: Orchestration_finalize
    public :: Orchestration_executeTasks

    interface
        subroutine runtimeTask(tId, work) bind(c)
            import
            implicit none
            integer(c_int), intent(IN), value :: tId
            integer(c_int), intent(IN)        :: work
        end subroutine runtimeTask
    end interface

contains

    !>
    !!
    !!
    subroutine Orchestration_init(nTeams, nThreadsPerTeam, logFilename)
        interface
            subroutine orchestration_init_fi(nTeams, &
                                             nThreadsPerTeam, &
                                             logFilename) bind(c)
                import
                implicit none
                integer,                      intent(IN), value :: nTeams
                integer,                      intent(IN), value :: nThreadsPerTeam
                character(len=1,kind=c_char), intent(IN)        :: logFilename(*)
            end subroutine orchestration_init_fi
        end interface

        integer,          intent(IN) :: nTeams
        integer,          intent(IN) :: nThreadsPerTeam
        character(len=*), intent(IN) :: logFilename 

        character(len=1,kind=c_char) :: cFilename(len_trim(logFilename)+1)
        integer                :: i

        if (isInitialized) then
            ! TODO: Add in stderr mechanism and write errors to this
            write(*,*) "The Orchestration Runtime has already been initialized" 
            STOP
        end if

        do i = 1, len_trim(logFilename)
           cFilename(i) = logFilename(i:i)
        end do
        cFilename(SIZE(cFilename)) = c_null_char

        call orchestration_init_fi(nTeams, nThreadsPerTeam, cFilename)
        isInitialized = .TRUE.
    end subroutine Orchestration_init

    !>
    !!
    !!
    subroutine Orchestration_finalize()
        interface
            subroutine orchestration_finalize_fi() bind(c)
                import
                implicit none
            end subroutine orchestration_finalize_fi
        end interface

        if (.NOT. isInitialized) then
            ! TODO: Add in stderr mechanism and write errors to this
            write(*,*) "The Orchestration Runtime has not been initialized" 
            STOP
        end if

        call orchestration_finalize_fi()
        isInitialized = .FALSE.
    end subroutine Orchestration_finalize

    !>
    !!
    !!
    subroutine Orchestration_executeTasks(cpuTask)
        interface
            subroutine orchestration_execute_tasks_fi(cpuTask) bind(c)
                import
                implicit none
                type(c_funptr), intent(IN), value :: cpuTask
            end subroutine orchestration_execute_tasks_fi
        end interface
                
        procedure(runtimeTask) :: cpuTask
        type(c_funptr)         :: cpuTaskPtr

        cpuTaskPtr = c_funloc(cpuTask)
        call orchestration_execute_tasks_fi(cpuTaskPtr)
    end subroutine Orchestration_executeTasks

end module Orchestration_runtime_mod

