!>
!!
!!
subroutine Orchestration_finalize()
    use Orchestration_data, ONLY : or_isRuntimeInitialized
    implicit none

    interface
        subroutine orchestration_finalize_fi() bind(c)
            use iso_c_binding
            implicit none
        end subroutine orchestration_finalize_fi
    end interface

    if (.NOT. or_isRuntimeInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Orchestration Runtime has not been initialized" 
        STOP
    end if

    CALL orchestration_finalize_fi()
    or_isRuntimeInitialized = .FALSE.
end subroutine Orchestration_finalize

