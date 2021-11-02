module hy_cg_interface
    !!!!!----- ROUTINES WRITTEN BY CODE GENERATOR
    ! Task-specific static Fortran layer
    interface
        subroutine hy_advanceSolutionHll_3_packet_oacc(dataQ_h,                &
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
            real,                          intent(OUT)   :: auxC_all_d(:,:,:,:)
            real,                          intent(OUT)   :: faceX_all_d(:,:,:,:,:)
            real,                          intent(OUT)   :: faceY_all_d(:,:,:,:,:)
            real,                          intent(OUT)   :: faceZ_all_d(:,:,:,:,:)
        end subroutine hy_advanceSolutionHll_3_packet_oacc

        subroutine hy_computeSoundSpeedHll_oacc(lo, hi, U, auxC)
            implicit none
            integer, intent(IN)  :: lo(:)
            integer, intent(IN)  :: hi(:)
            real,    intent(IN)  :: U(:,:,:,:)
            real,    intent(OUT) :: auxC(:,:,:)
        end subroutine hy_computeSoundSpeedHll_oacc

        subroutine hy_computeFluxesHll_X_oacc(dt, lo, hi, deltas, U, auxC, flX)
            implicit none
            real,    intent(IN)  :: dt
            integer, intent(IN)  :: lo(:)
            integer, intent(IN)  :: hi(:)
            real,    intent(IN)  :: deltas(:)
            real,    intent(IN)  :: U(:,:,:,:)
            real,    intent(IN)  :: auxC(:,:,:)
            real,    intent(OUT) :: flX(:,:,:,:)
        end subroutine hy_computeFluxesHll_X_oacc

        subroutine hy_computeFluxesHll_Y_oacc(dt, lo, hi, deltas, U, auxC, flY)
            implicit none
            real,    intent(IN)  :: dt
            integer, intent(IN)  :: lo(:)
            integer, intent(IN)  :: hi(:)
            real,    intent(IN)  :: deltas(:)
            real,    intent(IN)  :: U(:,:,:,:)
            real,    intent(IN)  :: auxC(:,:,:)
            real,    intent(OUT) :: flY(:,:,:,:)
        end subroutine hy_computeFluxesHll_Y_oacc

        subroutine hy_computeFluxesHll_Z_oacc(dt, lo, hi, deltas, U, auxC, flZ)
            implicit none
            real,    intent(IN)  :: dt
            integer, intent(IN)  :: lo(:)
            integer, intent(IN)  :: hi(:)
            real,    intent(IN)  :: deltas(:)
            real,    intent(IN)  :: U(:,:,:,:)
            real,    intent(IN)  :: auxC(:,:,:)
            real,    intent(OUT) :: flZ(:,:,:,:)
        end subroutine hy_computeFluxesHll_Z_oacc

        subroutine hy_updateSolutionHll_oacc(lo, hi, flX, flY, flZ, U)
            implicit none
            integer, intent(IN)    :: lo(:)
            integer, intent(IN)    :: hi(:)
            real,    intent(IN)    :: flX(:,:,:,:)
            real,    intent(IN)    :: flY(:,:,:,:)
            real,    intent(IN)    :: flZ(:,:,:,:)
            real,    intent(INOUT) :: U(:,:,:,:)
        end subroutine hy_updateSolutionHll_oacc
    end interface

end module hy_cg_interface

