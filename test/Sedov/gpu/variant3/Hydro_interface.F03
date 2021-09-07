module Hydro_interface
    !!!!!----- ROUTINES WRITTEN BY CODE GENERATOR
    ! Task-specific static Fortran layer
    interface
        subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h,            &
                                                          nTiles_d, dt_d,     &
                                                          deltas_all_d,       &
                                                          lo_all_d, hi_all_d)
            use openacc, ONLY : acc_handle_kind
            implicit none
            integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
            integer,                       intent(INOUT) :: nTiles_d
            real,                          intent(INOUT) :: dt_d
            real,                          intent(INOUT) :: deltas_all_d(:, :)
            integer,                       intent(INOUT) :: lo_all_d(:, :)
            integer,                       intent(INOUT) :: hi_all_d(:, :)
        end subroutine Hydro_advanceSolutionHll_3_packet_oacc
    end interface
end module Hydro_interface

