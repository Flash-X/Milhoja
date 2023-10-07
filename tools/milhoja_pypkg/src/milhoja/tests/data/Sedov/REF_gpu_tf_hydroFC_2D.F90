module gpu_tf_hydroFC_mod
    implicit none
    private

    public :: gpu_tf_hydroFC

contains

    subroutine gpu_tf_hydroFC(       &
                    C_packet_h,      &
                    dataQ_h,         &
                    nTiles_d,        &
                    dt_d,            &
                    tile_lo_d,       &
                    tile_hi_d,       &
                    tile_deltas_d,   &
                    CC_1_d,          &
                    FLX_1_d,         &
                    FLY_1_d,         &
                    FLZ_1_d,         &
                    hydro_op1_auxc_d &
            )
        use iso_c_binding, ONLY : C_PTR
        use openacc

        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeSoundSpeedHll_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_X_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Y_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_updateSolutionHll_gpu_oacc

        !$acc routine (Hydro_computeSoundSpeedHll_gpu_oacc) vector
        !$acc routine (Hydro_computeFluxesHll_X_gpu_oacc) vector
        !$acc routine (Hydro_computeFluxesHll_Y_gpu_oacc) vector
        !$acc routine (Hydro_updateSolutionHll_gpu_oacc) vector

        type(C_PTR),                   intent(IN)    :: C_packet_h
        integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
        integer,                       intent(IN)    :: nTiles_d
        real,                          intent(IN)    :: dt_d
        integer,                       intent(IN)    :: tile_lo_d(:, :)
        integer,                       intent(IN)    :: tile_hi_d(:, :)
        real,                          intent(IN)    :: tile_deltas_d(:, :)
        real,                          intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLX_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLY_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLZ_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_auxc_d(:, :, :, :)

        integer :: n

        !$acc data                   &
        !$acc& deviceptr(            &
        !$acc&      nTiles_d,        &
        !$acc&      dt_d,            &
        !$acc&      tile_lo_d,       &
        !$acc&      tile_hi_d,       &
        !$acc&      tile_deltas_d,   &
        !$acc&      CC_1_d,          &
        !$acc&      FLX_1_d,         &
        !$acc&      FLY_1_d,         &
        !$acc&      FLZ_1_d,         &
        !$acc&      hydro_op1_auxc_d &
        !$acc&  )

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_computeSoundSpeedHll_gpu_oacc( &
                    tile_lo_d,                        &
                    tile_hi_d,                        &
                    CC_1_d,                           &
                    hydro_op1_auxc_d                  &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_computeFluxesHll_X_gpu_oacc( &
                    dt_d,                           &
                    tile_lo_d,                      &
                    tile_hi_d,                      &
                    tile_deltas_d,                  &
                    CC_1_d,                         &
                    hydro_op1_auxc_d,               &
                    FLX_1_d                         &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_computeFluxesHll_Y_gpu_oacc( &
                    dt_d,                           &
                    tile_lo_d,                      &
                    tile_hi_d,                      &
                    tile_deltas_d,                  &
                    CC_1_d,                         &
                    hydro_op1_auxc_d,               &
                    FLY_1_d                         &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_updateSolutionHll_gpu_oacc( &
                    tile_lo_d,                     &
                    tile_hi_d,                     &
                    FLX_1_d,                       &
                    FLY_1_d,                       &
                    FLZ_1_d,                       &
                    CC_1_d                         &
                 )
        end do
        !$acc end parallel loop

        !$acc wait(dataQ_h)

        !$acc end data
    end subroutine gpu_tf_hydroFC

end module gpu_tf_hydroFC_mod
