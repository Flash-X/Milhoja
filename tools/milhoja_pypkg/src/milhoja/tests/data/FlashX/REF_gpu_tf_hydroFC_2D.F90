module gpu_tf_hydroFC_mod
    implicit none
    private

    public :: gpu_tf_hydroFC_Fortran
    public :: gpu_tf_hydroFC_Cpp2C

    interface
        !> C++ task function that TimeAdvance passes to Orchestration unit
        subroutine gpu_tf_hydroFC_Cpp2C(C_tId, C_dataItemPtr) &
                bind(c, name="gpu_tf_hydroFC_Cpp2C")
            use iso_c_binding, ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            integer(MILHOJA_INT), intent(IN), value :: C_tId
            type(C_PTR), intent(IN), value :: C_dataItemPtr
        end subroutine gpu_tf_hydroFC_Cpp2C
    end interface

contains

    subroutine gpu_tf_hydroFC_Fortran(       &
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
        use openacc, ONLY : acc_handle_kind

        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeSoundSpeedHll_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_X_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Y_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_updateSolutionHll_gpu_oacc

#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (wrapper_Hydro_computeSoundSpeedHll_gpu_oacc) vector
#endif
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (wrapper_Hydro_computeFluxesHll_X_gpu_oacc) vector
#endif
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (wrapper_Hydro_computeFluxesHll_Y_gpu_oacc) vector
#endif
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (wrapper_Hydro_updateSolutionHll_gpu_oacc) vector
#endif

        implicit none

        type(C_PTR),                   intent(IN)    :: C_packet_h
        integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
        integer,                       intent(IN)    :: nTiles_d
        real,                          intent(IN)    :: dt_d
        integer,                       intent(IN)    :: tile_lo_d(:, :)
        integer,                       intent(IN)    :: tile_hi_d(:, :)
        real,                          intent(IN)    :: tile_deltas_d(:, :)
        real,                          intent(INOUT)   :: CC_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLX_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLY_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: FLZ_1_d(:, :, :, :, :)
        real,                          intent(INOUT)   :: hydro_op1_auxc_d(:, :, :, :)

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
            CALL wrapper_Hydro_computeSoundSpeedHll_gpu_oacc( &
                    n, &
                    tile_lo_d, &
                    tile_hi_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d &
                    )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeFluxesHll_X_gpu_oacc( &
                    n, &
                    dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d, &
                    FLX_1_d &
                    )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeFluxesHll_Y_gpu_oacc( &
                    n, &
                    dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d, &
                    FLY_1_d &
                    )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_updateSolutionHll_gpu_oacc( &
                    n, &
                    tile_lo_d, &
                    tile_hi_d, &
                    FLX_1_d, &
                    FLY_1_d, &
                    FLZ_1_d, &
                    CC_1_d &
                    )
        end do
        !$acc end parallel loop

        !$acc wait( &
        !$acc&      dataQ_h &
        !$acc&  )

        !$acc end data
    end subroutine gpu_tf_hydroFC_Fortran

    subroutine wrapper_Hydro_computeSoundSpeedHll_gpu_oacc ( &
                    nblk, &
                    tile_lo_d, &
                    tile_hi_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeSoundSpeedHll_gpu_oacc

        !$acc routine vector
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (Hydro_computeSoundSpeedHll_gpu_oacc) vector
#endif

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real, target, intent(INOUT) :: hydro_op1_auxc_d(:, :, :, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        real, pointer :: hydro_op1_auxc_d_p(:, :, :)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        hydro_op1_auxc_d_p => hydro_op1_auxc_d(:, :, :, nblk)

        ! Call subroutine
        CALL Hydro_computeSoundSpeedHll_gpu_oacc( &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    CC_1_d_p, &
                    hydro_op1_auxc_d_p &
                )

    end subroutine wrapper_Hydro_computeSoundSpeedHll_gpu_oacc

    subroutine wrapper_Hydro_computeFluxesHll_X_gpu_oacc ( &
                    nblk, &
                    dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d, &
                    FLX_1_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeFluxesHll_X_gpu_oacc

        !$acc routine vector
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (Hydro_computeFluxesHll_X_gpu_oacc) vector
#endif

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        real, target, intent(IN) :: dt_d
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(IN) :: tile_deltas_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real, target, intent(INOUT) :: hydro_op1_auxc_d(:, :, :, :)
        real, target, intent(OUT) :: FLX_1_d(:, :, :, :, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: tile_deltas_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        real, pointer :: hydro_op1_auxc_d_p(:, :, :)
        real, pointer :: FLX_1_d_p(:, :, :, :)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        tile_deltas_d_p => tile_deltas_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        hydro_op1_auxc_d_p => hydro_op1_auxc_d(:, :, :, nblk)
        FLX_1_d_p => FLX_1_d(:, :, :, :, nblk)

        ! Call subroutine
        CALL Hydro_computeFluxesHll_X_gpu_oacc( &
                    dt_d, &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    tile_deltas_d_p, &
                    CC_1_d_p, &
                    hydro_op1_auxc_d_p, &
                    FLX_1_d_p &
                )

    end subroutine wrapper_Hydro_computeFluxesHll_X_gpu_oacc

    subroutine wrapper_Hydro_computeFluxesHll_Y_gpu_oacc ( &
                    nblk, &
                    dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    hydro_op1_auxc_d, &
                    FLY_1_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeFluxesHll_Y_gpu_oacc

        !$acc routine vector
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (Hydro_computeFluxesHll_Y_gpu_oacc) vector
#endif

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        real, target, intent(IN) :: dt_d
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(IN) :: tile_deltas_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real, target, intent(INOUT) :: hydro_op1_auxc_d(:, :, :, :)
        real, target, intent(OUT) :: FLY_1_d(:, :, :, :, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: tile_deltas_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        real, pointer :: hydro_op1_auxc_d_p(:, :, :)
        real, pointer :: FLY_1_d_p(:, :, :, :)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        tile_deltas_d_p => tile_deltas_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        hydro_op1_auxc_d_p => hydro_op1_auxc_d(:, :, :, nblk)
        FLY_1_d_p => FLY_1_d(:, :, :, :, nblk)

        ! Call subroutine
        CALL Hydro_computeFluxesHll_Y_gpu_oacc( &
                    dt_d, &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    tile_deltas_d_p, &
                    CC_1_d_p, &
                    hydro_op1_auxc_d_p, &
                    FLY_1_d_p &
                )

    end subroutine wrapper_Hydro_computeFluxesHll_Y_gpu_oacc

    subroutine wrapper_Hydro_updateSolutionHll_gpu_oacc ( &
                    nblk, &
                    tile_lo_d, &
                    tile_hi_d, &
                    FLX_1_d, &
                    FLY_1_d, &
                    FLZ_1_d, &
                    CC_1_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_updateSolutionHll_gpu_oacc

        !$acc routine vector
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
        !$acc routine (Hydro_updateSolutionHll_gpu_oacc) vector
#endif

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(OUT) :: FLX_1_d(:, :, :, :, :)
        real, target, intent(OUT) :: FLY_1_d(:, :, :, :, :)
        real, target, intent(OUT) :: FLZ_1_d(:, :, :, :, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: FLX_1_d_p(:, :, :, :)
        real, pointer :: FLY_1_d_p(:, :, :, :)
        real, pointer :: FLZ_1_d_p(:, :, :, :)
        real, pointer :: CC_1_d_p(:, :, :, :)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        FLX_1_d_p => FLX_1_d(:, :, :, :, nblk)
        FLY_1_d_p => FLY_1_d(:, :, :, :, nblk)
        FLZ_1_d_p => FLZ_1_d(:, :, :, :, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)

        ! Call subroutine
        CALL Hydro_updateSolutionHll_gpu_oacc( &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    FLX_1_d_p, &
                    FLY_1_d_p, &
                    FLZ_1_d_p, &
                    CC_1_d_p &
                )

    end subroutine wrapper_Hydro_updateSolutionHll_gpu_oacc

end module gpu_tf_hydroFC_mod
