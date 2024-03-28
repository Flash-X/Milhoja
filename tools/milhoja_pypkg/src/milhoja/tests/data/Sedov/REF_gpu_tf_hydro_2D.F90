module gpu_tf_hydro_mod
    implicit none
    private

    public :: gpu_tf_hydro

    ! NOTE TO MILHOJA USERS:
    ! The Fortran interfaces defined here should be used to create and manage
    ! prototype data items to be passed to Milhoja for use with this task
    ! function.
    interface
        ! Instantiate the prototype data packet
        !
        ! Arguments
        !   C_packet - Milhoja-internal handle to the data packet
        !   All others - user-specified external arguments
        function instantiate_gpu_tf_hydro_packet_C( &
                    C_packet, &
                    C_dt &
                ) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            use milhoja_types_mod, ONLY : MILHOJA_REAL
            type(C_PTR),         intent(IN)        :: C_packet
            real(MILHOJA_REAL),  intent(IN), value :: C_dt
            integer(MILHOJA_INT)                   :: C_ierr
        end function instantiate_gpu_tf_hydro_packet_C

        ! Delete the prototype data packet
        !
        ! Arguments
        !   C_packet - Milhoja-internal handle obtained when instantiating
        !               data packet
        function delete_gpu_tf_hydro_packet_C(C_packet) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            type(C_PTR),         intent(IN), value :: C_packet
            integer(MILHOJA_INT)                   :: C_ierr
        end function delete_gpu_tf_hydro_packet_C
    end interface

contains

    subroutine gpu_tf_hydro(         &
                    C_packet_h,      &
                    dataQ_h,         &
                    nTiles_d,        &
                    dt_d,            &
                    tile_lo_d,       &
                    tile_hi_d,       &
                    tile_deltas_d,   &
                    CC_1_d,          &
                    hydro_op1_flX_d, &
                    hydro_op1_flY_d, &
                    hydro_op1_flZ_d, &
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
        real,                          intent(OUT)   :: hydro_op1_flX_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_flY_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_flZ_d(:, :, :, :, :)
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
        !$acc&      hydro_op1_flX_d, &
        !$acc&      hydro_op1_flY_d, &
        !$acc&      hydro_op1_flZ_d, &
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
                    hydro_op1_flX_d                 &
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
                    hydro_op1_flY_d                 &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_updateSolutionHll_gpu_oacc( &
                    tile_lo_d,                     &
                    tile_hi_d,                     &
                    hydro_op1_flX_d,               &
                    hydro_op1_flY_d,               &
                    hydro_op1_flZ_d,               &
                    CC_1_d                         &
                 )
        end do
        !$acc end parallel loop

        !$acc wait( &
        !$acc&      dataQ_h &
        !$acc&  )

        !$acc end data
    end subroutine gpu_tf_hydro

end module gpu_tf_hydro_mod
