!>
!!
!!
subroutine Orchestration_init(nTeams, nThreadsPerTeam, logFilename)
    use iso_c_binding,      ONLY : c_char, c_null_char
    use Orchestration_data, ONLY : isRuntimeInitialized
    implicit none

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

    character(len=1,kind=c_char) :: cFilename(LEN_TRIM(logFilename)+1)
    integer                      :: i

    if (isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has already been initialized" 
        STOP
    end if

    do i = 1, LEN_TRIM(logFilename)
       cFilename(i) = logFilename(i:i)
    end do
    cFilename(SIZE(cFilename)) = c_null_char

    CALL orchestration_init_fi(nTeams, nThreadsPerTeam, cFilename)
    isRuntimeInitialized = .TRUE.
end subroutine Orchestration_init

