!>
!!
!!
subroutine Orchestration_init()
    use iso_c_binding,      ONLY : C_CHAR, C_NULL_CHAR
    use Orchestration_data, ONLY : or_isRuntimeInitialized, &
                                   or_logFilename, &
                                   or_nThreadTeams, &
                                   or_nThreadsPerTeam
    implicit none

    interface
        subroutine orchestration_init_fi(nTeams, &
                                         nThreadsPerTeam, &
                                         logFilename) bind(c)
            import
            implicit none
            integer,                      intent(IN), value :: nTeams
            integer,                      intent(IN), value :: nThreadsPerTeam
            character(len=1,kind=C_CHAR), intent(IN)        :: logFilename(*)
        end subroutine orchestration_init_fi
    end interface

    character(len=1,kind=C_CHAR) :: cFilename(LEN_TRIM(or_logFilename)+1)
    integer                      :: i

    if (or_isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has already been initialized" 
        STOP
    end if

    do i = 1, LEN_TRIM(or_logFilename)
       cFilename(i) = or_logFilename(i:i)
    end do
    cFilename(SIZE(cFilename)) = C_NULL_CHAR

    CALL orchestration_init_fi(or_nThreadTeams, or_nThreadsPerTeam, cFilename)
    or_isRuntimeInitialized = .TRUE.
end subroutine Orchestration_init

