!>
!!
!!
subroutine Grid_initDomain(xMin, xMax, yMin, yMax, zMin, zMax, &
                           nBlocksX, nBlocksY, nBlocksZ, nVars)
    use iso_c_binding, ONLY : C_INT, C_DOUBLE, C_FUNLOC

    use Simulation_interface, ONLY : Simulation_initBlock
    use Grid_interface,       ONLY : Grid_getDomainBoundBox
    use Grid_data,            ONLY : gr_isGridInitialized, &
                                     gr_globalDomain

    implicit none

    interface
        subroutine grid_init_domain_fi(xMin, xMax, yMin, yMax, zMin, zMax, &
                                       nBlocksX, nBlocksY, nBlocksZ, nVars, &
                                       initBlock) bind(c)
            use iso_c_binding, ONLY : C_INT, C_DOUBLE, C_FUNPTR
            implicit none
            real(C_DOUBLE), intent(IN), value :: xMin
            real(C_DOUBLE), intent(IN), value :: xMax
            real(C_DOUBLE), intent(IN), value :: yMin
            real(C_DOUBLE), intent(IN), value :: yMax
            real(C_DOUBLE), intent(IN), value :: zMin
            real(C_DOUBLE), intent(IN), value :: zMax
            integer(C_INT), intent(IN), value :: nBlocksX
            integer(C_INT), intent(IN), value :: nBlocksY
            integer(C_INT), intent(IN), value :: nBlocksZ
            integer(C_INT), intent(IN), value :: nVars
            type(C_FUNPTR), intent(IN), value :: initBlock
        end subroutine grid_init_domain_fi
    end interface

    real,    intent(IN)   :: xMin
    real,    intent(IN)   :: xMax
    real,    intent(IN)   :: yMin
    real,    intent(IN)   :: yMax
    real,    intent(IN)   :: zMin
    real,    intent(IN)   :: zMax
    integer, intent(IN)   :: nBlocksX
    integer, intent(IN)   :: nBlocksY
    integer, intent(IN)   :: nBlocksZ
    integer, intent(IN)   :: nVars

    if (.NOT. gr_isGridInitialized) then
        write(*,*) "The Grid has not been initialized" 
        STOP
    end if

    CALL grid_init_domain_fi(REAL(xMin, C_DOUBLE), REAL(xMax, C_DOUBLE), &
                             REAL(yMin, C_DOUBLE), REAL(yMax, C_DOUBLE), &
                             REAL(zMin, C_DOUBLE), REAL(zMax, C_DOUBLE), &
                             INT(nBlocksX, C_INT), &
                             INT(nBlocksY, C_INT), &
                             INT(nBlocksZ, C_INT), &
                             INT(nVars,    C_INT), &
                             C_FUNLOC(Simulation_initBlock))

    CALL Grid_getDomainBoundBox(gr_globalDomain)
end subroutine Grid_initDomain

