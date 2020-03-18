!>
!!
!!
subroutine Grid_init()
    use Grid_data, ONLY : gr_isGridInitialized

    implicit none

    interface
        subroutine grid_init_fi() bind(c)
            implicit none
        end subroutine grid_init_fi
    end interface

    if (gr_isGridInitialized) then
        ! TODO: Add in stderr mechanism and write errors to this
        write(*,*) "The Grid has already been initialized" 
        STOP
    end if

    CALL grid_init_fi()
    gr_isGridInitialized = .TRUE.
end subroutine Grid_init

