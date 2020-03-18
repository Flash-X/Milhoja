!>
!!
!!
subroutine Grid_finalize()
    use Grid_data, ONLY : gr_isGridInitialized
    implicit none

    interface
        subroutine grid_finalize_fi() bind(c)
            implicit none
        end subroutine grid_finalize_fi
    end interface

    if (.NOT. gr_isGridInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    CALL grid_finalize_fi()
    gr_isGridInitialized = .FALSE.
end subroutine Grid_finalize

