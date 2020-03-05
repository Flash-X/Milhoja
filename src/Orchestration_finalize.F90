!>
!!
!!
subroutine Orchestration_finalize()
    use Orchestration_data, ONLY : isRuntimeInitialized
    implicit none

    interface
        subroutine orchestration_finalize_fi() bind(c)
            implicit none
        end subroutine orchestration_finalize_fi
    end interface

    if (.NOT. isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has not been initialized" 
        STOP
    end if

    CALL orchestration_finalize_fi()
    isRuntimeInitialized = .FALSE.
end subroutine Orchestration_finalize

