module Hydro_interface
    !!!!!----- ROUTINES WRITTEN BY CODE GENERATOR
    ! Task-specific static Fortran layer
    interface
        subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h,                &
                                                          nTiles_d, dt_d,         &
                                                          deltas_all_d,           &
                                                          lo_all_d,   hi_all_d,   &
                                                          loGC_all_d, hiGC_all_d, &
                                                          U_all_d,                &
                                                          auxC_all_d,             &
                                                          faceX_all_d,            &
                                                          faceY_all_d,            &
                                                          faceZ_all_d)
            use openacc, ONLY : acc_handle_kind
            implicit none
            integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
            integer,                       intent(IN)    :: nTiles_d
            real,                          intent(IN)    :: dt_d
            real,                          intent(IN)    :: deltas_all_d(:,:)
            integer,                       intent(IN)    :: lo_all_d(:,:)
            integer,                       intent(IN)    :: hi_all_d(:,:)
            integer,                       intent(IN)    :: loGC_all_d(:,:)
            integer,                       intent(IN)    :: hiGC_all_d(:,:)
            real,                          intent(INOUT) :: U_all_d(:,:,:,:,:)
            real,                          intent(INOUT) :: auxC_all_d(:,:,:,:)
            real,                          intent(INOUT) :: faceX_all_d(:,:,:,:,:)
            real,                          intent(INOUT) :: faceY_all_d(:,:,:,:,:)
            real,                          intent(INOUT) :: faceZ_all_d(:,:,:,:,:)
        end subroutine Hydro_advanceSolutionHll_3_packet_oacc
    end interface
end module Hydro_interface

