!>
!!
!!
subroutine Orchestration_init()
    use iso_c_binding,      ONLY : C_INT, C_LONG_LONG, C_CHAR, C_NULL_CHAR
    use Orchestration_data, ONLY : or_isRuntimeInitialized, &
                                   or_logFilename, &
                                   or_nThreadTeams, &
                                   or_nThreadsPerTeam, &
                                   or_nStreams, &
                                   or_nBytesInMemoryPools

    implicit none

    interface
        function orchestration_init_fi(nTeams, &
                                       nThreadsPerTeam, &
                                       nStreams, &
                                       nBytesInMemoryPools, &
                                       logFilename) result(success) bind(c)
            import
            implicit none
            integer(C_INT),               intent(IN), value :: nTeams
            integer(C_INT),               intent(IN), value :: nThreadsPerTeam
            integer(C_INT),               intent(IN), value :: nStreams
            integer(C_LONG_LONG),         intent(IN), value :: nBytesInMemoryPools
            character(len=1,kind=C_CHAR), intent(IN)        :: logFilename(*)
            integer(C_INT)                                  :: success
        end function orchestration_init_fi
    end interface

    character(len=1,kind=C_CHAR) :: cFilename(LEN_TRIM(or_logFilename)+1)
    integer(C_INT)               :: success
    integer                      :: i

    if (or_isRuntimeInitialized) then
        write(*,*) "The Orchestration Runtime has already been initialized" 
        STOP
    end if

    do i = 1, LEN_TRIM(or_logFilename)
       cFilename(i) = or_logFilename(i:i)
    end do
    cFilename(SIZE(cFilename)) = C_NULL_CHAR

    success = orchestration_init_fi(or_nThreadTeams, &
                                    or_nThreadsPerTeam, &
                                    or_nStreams, &
                                    or_nBytesInMemoryPools, &
                                    cFilename)
    if (success /= 1) then
        write(*,*) "[Orchestration_init] Unable to initialize orchestration runtime"
        STOP
    end if

    or_isRuntimeInitialized = .TRUE.
end subroutine Orchestration_init

