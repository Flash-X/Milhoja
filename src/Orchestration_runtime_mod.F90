module Orchestration_runtime_mod 
    use iso_c_binding

    implicit none
    private

    logical, save :: isInitialized = .FALSE.

    public :: Orchestration_init
!    public :: Orchestration_executeTasks
    public :: Orchestration_finalize

contains

    subroutine Orchestration_init(nTeams, nThreadsPerTeam, logFilename)
        interface
            subroutine Orchestration_init_fi(nTeams, &
                                             nThreadsPerTeam, &
                                             logFilename) bind(c)
                import
                implicit none
                integer,                intent(IN), value :: nTeams
                integer,                intent(IN), value :: nThreadsPerTeam
                character(kind=c_char), intent(IN)        :: logFilename(*)
            end subroutine Orchestration_init_fi
        end interface

        integer,      intent(IN) :: nTeams
        integer,      intent(IN) :: nThreadsPerTeam
        character(*), intent(IN) :: logFilename 

        character(kind=c_char) :: cFilename(len_trim(logFilename)+1)
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

        call Orchestration_init_fi(nTeams, nThreadsPerTeam, cFilename)
        isInitialized = .TRUE.
    end subroutine Orchestration_init

    subroutine Orchestration_finalize()
        interface
            subroutine Orchestration_finalize_fi() bind(c)
                import
                implicit none
            end subroutine Orchestration_finalize_fi
        end interface

        if (.NOT. isInitialized) then
            ! TODO: Add in stderr mechanism and write errors to this
            write(*,*) "The Orchestration Runtime has not been initialized" 
            STOP
        end if

        call Orchestration_finalize_fi()
        isInitialized = .FALSE.
    end subroutine Orchestration_finalize

end module Orchestration_runtime_mod

