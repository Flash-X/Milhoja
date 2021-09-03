module Hydro_interface
    !!!!!----- ROUTINES WRITTEN BY CODE GENERATOR
    ! Task-specific static Fortran layer
    interface
        subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h, &
                                                          nTiles_d)
            use openacc, ONLY : acc_handle_kind
            implicit none
            integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
            integer,                       intent(INOUT) :: nTiles_d
        end subroutine Hydro_advanceSolutionHll_3_packet_oacc
    end interface
end module Hydro_interface

