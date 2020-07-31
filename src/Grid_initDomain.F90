!>
!!
!!
subroutine Grid_initDomain()
    use iso_c_binding, ONLY : C_INT, C_DOUBLE, C_FUNLOC

    use Simulation_interface, ONLY : Simulation_initBlock
    use Grid_interface,       ONLY : Grid_getDomainBoundBox
    use Grid_data,            ONLY : gr_isGridInitialized, &
                                     gr_globalDomain

    implicit none

    interface
        subroutine grid_init_domain_fi(initBlock) bind(c)
            use iso_c_binding, ONLY : C_FUNPTR
            implicit none
            type(C_FUNPTR), intent(IN), value :: initBlock
        end subroutine grid_init_domain_fi
    end interface

    if (.NOT. gr_isGridInitialized) then
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    CALL grid_init_domain_fi(C_FUNLOC(Simulation_initBlock))

    CALL Grid_getDomainBoundBox(gr_globalDomain)
end subroutine Grid_initDomain

