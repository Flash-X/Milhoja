#include "Milhoja_interface_error_codes.h"

module gpu_tf_hydro_mod
    implicit none
    private

    public :: gpu_tf_hydro_Fortran
    public :: gpu_tf_hydro_Cpp2C

    interface
        !> C++ task function that TimeAdvance passes to Orchestration unit
        subroutine gpu_tf_hydro_Cpp2C(C_tId, C_dataItemPtr) &
                bind(c, name="gpu_tf_hydro_Cpp2C")
            use iso_c_binding, ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            integer(MILHOJA_INT), intent(IN), value :: C_tId
            type(C_PTR), intent(IN), value :: C_dataItemPtr
        end subroutine gpu_tf_hydro_Cpp2C
    end interface

contains

    subroutine gpu_tf_hydro_Fortran( &
                    C_packet_h, &
                    dataQ_h, &
                    queue2_h, &
                    queue3_h, &
                    nTiles_d, &
                    external_hydro_op1_dt_d, &
                    tile_deltas_d, &
                    tile_hi_d, &
                    tile_lo_d, &
                    CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flX_d, &
                    scratch_hydro_op1_flY_d, &
                    scratch_hydro_op1_flZ_d, &
                    lbdd_CC_1_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_flX_d, &
                    lbdd_scratch_hydro_op1_flY_d, &
                    lbdd_scratch_hydro_op1_flZ_d &
            )
        use DataPacket_gpu_tf_hydro_c2f_mod, ONLY : release_gpu_tf_hydro_extra_queue_c
        use iso_c_binding, ONLY : C_PTR
        use openacc

        use milhoja_types_mod, ONLY : MILHOJA_INT

        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeSoundSpeedHll_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_X_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Y_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Z_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_updateSolutionHll_gpu_oacc

        !$acc routine (wrapper_Hydro_computeSoundSpeedHll_gpu_oacc) vector
        !$acc routine (wrapper_Hydro_computeFluxesHll_X_gpu_oacc) vector
        !$acc routine (wrapper_Hydro_computeFluxesHll_Y_gpu_oacc) vector
        !$acc routine (wrapper_Hydro_computeFluxesHll_Z_gpu_oacc) vector
        !$acc routine (wrapper_Hydro_updateSolutionHll_gpu_oacc) vector

        implicit none

        type(C_PTR), intent(IN) :: C_packet_h
        integer(kind=acc_handle_kind), intent(IN) :: dataQ_h
        integer(kind=acc_handle_kind), intent(IN) :: queue2_h
        integer(kind=acc_handle_kind), intent(IN) :: queue3_h
        integer, intent(IN) :: nTiles_d
        real, intent(IN) :: external_hydro_op1_dt_d
        real, intent(IN) :: tile_deltas_d(:, :)
        integer, intent(IN) :: tile_hi_d(:, :)
        integer, intent(IN) :: tile_lo_d(:, :)
        real, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real, intent(INOUT) :: scratch_hydro_op1_auxC_d(:, :, :, :)
        real, intent(INOUT) :: scratch_hydro_op1_flX_d(:, :, :, :, :)
        real, intent(INOUT) :: scratch_hydro_op1_flY_d(:, :, :, :, :)
        real, intent(INOUT) :: scratch_hydro_op1_flZ_d(:, :, :, :, :)
        integer, intent(IN) :: lbdd_CC_1_d(:, :)
        integer, intent(IN) :: lbdd_scratch_hydro_op1_auxC_d(:, :)
        integer, intent(IN) :: lbdd_scratch_hydro_op1_flX_d(:, :)
        integer, intent(IN) :: lbdd_scratch_hydro_op1_flY_d(:, :)
        integer, intent(IN) :: lbdd_scratch_hydro_op1_flZ_d(:, :)

        integer :: n

        integer(MILHOJA_INT) :: MH_idx
        integer(MILHOJA_INT) :: MH_ierr

        !$acc data &
        !$acc& deviceptr( &
        !$acc&        nTiles_d, &
        !$acc&        external_hydro_op1_dt_d, &
        !$acc&        tile_deltas_d, &
        !$acc&        tile_hi_d, &
        !$acc&        tile_lo_d, &
        !$acc&        CC_1_d, &
        !$acc&        scratch_hydro_op1_auxC_d, &
        !$acc&        scratch_hydro_op1_flX_d, &
        !$acc&        scratch_hydro_op1_flY_d, &
        !$acc&        scratch_hydro_op1_flZ_d, &
        !$acc&        lbdd_CC_1_d, &
        !$acc&        lbdd_scratch_hydro_op1_auxC_d, &
        !$acc&        lbdd_scratch_hydro_op1_flX_d, &
        !$acc&        lbdd_scratch_hydro_op1_flY_d, &
        !$acc&        lbdd_scratch_hydro_op1_flZ_d &
        !$acc&    )

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeSoundSpeedHll_gpu_oacc( &
                    n, &
                    tile_lo_d, &
                    tile_hi_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d &
                    )
        end do
        !$acc end parallel loop

        !$acc wait(dataQ_h)

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeFluxesHll_X_gpu_oacc( &
                    n, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flX_d, &
                    lbdd_scratch_hydro_op1_flX_d &
                    )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(queue2_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeFluxesHll_Y_gpu_oacc( &
                    n, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flY_d, &
                    lbdd_scratch_hydro_op1_flY_d &
                    )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(queue3_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_computeFluxesHll_Z_gpu_oacc( &
                    n, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flZ_d, &
                    lbdd_scratch_hydro_op1_flZ_d &
                    )
        end do
        !$acc end parallel loop

        !$acc wait( &
        !$acc&        queue2_h, &
        !$acc&        queue3_h &
        !$acc& )

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL wrapper_Hydro_updateSolutionHll_gpu_oacc( &
                    n, &
                    tile_lo_d, &
                    tile_hi_d, &
                    scratch_hydro_op1_flX_d, &
                    scratch_hydro_op1_flY_d, &
                    scratch_hydro_op1_flZ_d, &
                    lbdd_scratch_hydro_op1_flX_d, &
                    CC_1_d, &
                    lbdd_CC_1_d &
                    )
        end do
        !$acc end parallel loop

        !$acc wait( &
        !$acc&        dataQ_h &
        !$acc&    )

        MH_idx = INT(2, kind=MILHOJA_INT)
        MH_ierr = release_gpu_tf_hydro_extra_queue_c(C_packet_h, MH_idx)
        if (MH_ierr /= MILHOJA_SUCCESS) then
            write(*,*) "[gpu_tf_hydro] Unable to release extra OpenACC async queue 2"
            STOP
        end if

        MH_idx = INT(3, kind=MILHOJA_INT)
        MH_ierr = release_gpu_tf_hydro_extra_queue_c(C_packet_h, MH_idx)
        if (MH_ierr /= MILHOJA_SUCCESS) then
            write(*,*) "[gpu_tf_hydro] Unable to release extra OpenACC async queue 3"
            STOP
        end if

        !$acc end data
    end subroutine gpu_tf_hydro_Fortran

    subroutine wrapper_Hydro_computeSoundSpeedHll_gpu_oacc ( &
                    nblk, &
                    tile_lo_d, &
                    tile_hi_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeSoundSpeedHll_gpu_oacc

        !$acc routine vector
        !$acc routine (Hydro_computeSoundSpeedHll_gpu_oacc) vector

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_CC_1_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_auxC_d(:, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_auxC_d(:, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        integer, pointer :: lbdd_CC_1_d_p(:)
        real, pointer :: scratch_hydro_op1_auxC_d_p(:, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_auxC_d_p(:)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        lbdd_CC_1_d_p => lbdd_CC_1_d(:, nblk)
        scratch_hydro_op1_auxC_d_p => scratch_hydro_op1_auxC_d(:, :, :, nblk)
        lbdd_scratch_hydro_op1_auxC_d_p => lbdd_scratch_hydro_op1_auxC_d(:, nblk)

        ! Call subroutine
        CALL Hydro_computeSoundSpeedHll_gpu_oacc( &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    CC_1_d_p, &
                    lbdd_CC_1_d_p, &
                    scratch_hydro_op1_auxC_d_p, &
                    lbdd_scratch_hydro_op1_auxC_d_p &
                )

    end subroutine wrapper_Hydro_computeSoundSpeedHll_gpu_oacc

    subroutine wrapper_Hydro_computeFluxesHll_X_gpu_oacc ( &
                    nblk, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flX_d, &
                    lbdd_scratch_hydro_op1_flX_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeFluxesHll_X_gpu_oacc

        !$acc routine vector
        !$acc routine (Hydro_computeFluxesHll_X_gpu_oacc) vector

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        real, target, intent(IN) :: external_hydro_op1_dt_d
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(IN) :: tile_deltas_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_CC_1_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_auxC_d(:, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_auxC_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flX_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_flX_d(:, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: tile_deltas_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        integer, pointer :: lbdd_CC_1_d_p(:)
        real, pointer :: scratch_hydro_op1_auxC_d_p(:, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_auxC_d_p(:)
        real, pointer :: scratch_hydro_op1_flX_d_p(:, :, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_flX_d_p(:)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        tile_deltas_d_p => tile_deltas_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        lbdd_CC_1_d_p => lbdd_CC_1_d(:, nblk)
        scratch_hydro_op1_auxC_d_p => scratch_hydro_op1_auxC_d(:, :, :, nblk)
        lbdd_scratch_hydro_op1_auxC_d_p => lbdd_scratch_hydro_op1_auxC_d(:, nblk)
        scratch_hydro_op1_flX_d_p => scratch_hydro_op1_flX_d(:, :, :, :, nblk)
        lbdd_scratch_hydro_op1_flX_d_p => lbdd_scratch_hydro_op1_flX_d(:, nblk)

        ! Call subroutine
        CALL Hydro_computeFluxesHll_X_gpu_oacc( &
                    external_hydro_op1_dt_d, &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    tile_deltas_d_p, &
                    CC_1_d_p, &
                    lbdd_CC_1_d_p, &
                    scratch_hydro_op1_auxC_d_p, &
                    lbdd_scratch_hydro_op1_auxC_d_p, &
                    scratch_hydro_op1_flX_d_p, &
                    lbdd_scratch_hydro_op1_flX_d_p &
                )

    end subroutine wrapper_Hydro_computeFluxesHll_X_gpu_oacc

    subroutine wrapper_Hydro_computeFluxesHll_Y_gpu_oacc ( &
                    nblk, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flY_d, &
                    lbdd_scratch_hydro_op1_flY_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeFluxesHll_Y_gpu_oacc

        !$acc routine vector
        !$acc routine (Hydro_computeFluxesHll_Y_gpu_oacc) vector

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        real, target, intent(IN) :: external_hydro_op1_dt_d
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(IN) :: tile_deltas_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_CC_1_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_auxC_d(:, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_auxC_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flY_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_flY_d(:, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: tile_deltas_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        integer, pointer :: lbdd_CC_1_d_p(:)
        real, pointer :: scratch_hydro_op1_auxC_d_p(:, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_auxC_d_p(:)
        real, pointer :: scratch_hydro_op1_flY_d_p(:, :, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_flY_d_p(:)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        tile_deltas_d_p => tile_deltas_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        lbdd_CC_1_d_p => lbdd_CC_1_d(:, nblk)
        scratch_hydro_op1_auxC_d_p => scratch_hydro_op1_auxC_d(:, :, :, nblk)
        lbdd_scratch_hydro_op1_auxC_d_p => lbdd_scratch_hydro_op1_auxC_d(:, nblk)
        scratch_hydro_op1_flY_d_p => scratch_hydro_op1_flY_d(:, :, :, :, nblk)
        lbdd_scratch_hydro_op1_flY_d_p => lbdd_scratch_hydro_op1_flY_d(:, nblk)

        ! Call subroutine
        CALL Hydro_computeFluxesHll_Y_gpu_oacc( &
                    external_hydro_op1_dt_d, &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    tile_deltas_d_p, &
                    CC_1_d_p, &
                    lbdd_CC_1_d_p, &
                    scratch_hydro_op1_auxC_d_p, &
                    lbdd_scratch_hydro_op1_auxC_d_p, &
                    scratch_hydro_op1_flY_d_p, &
                    lbdd_scratch_hydro_op1_flY_d_p &
                )

    end subroutine wrapper_Hydro_computeFluxesHll_Y_gpu_oacc

    subroutine wrapper_Hydro_computeFluxesHll_Z_gpu_oacc ( &
                    nblk, &
                    external_hydro_op1_dt_d, &
                    tile_lo_d, &
                    tile_hi_d, &
                    tile_deltas_d, &
                    CC_1_d, &
                    lbdd_CC_1_d, &
                    scratch_hydro_op1_auxC_d, &
                    lbdd_scratch_hydro_op1_auxC_d, &
                    scratch_hydro_op1_flZ_d, &
                    lbdd_scratch_hydro_op1_flZ_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_computeFluxesHll_Z_gpu_oacc

        !$acc routine vector
        !$acc routine (Hydro_computeFluxesHll_Z_gpu_oacc) vector

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        real, target, intent(IN) :: external_hydro_op1_dt_d
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(IN) :: tile_deltas_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_CC_1_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_auxC_d(:, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_auxC_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flZ_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_flZ_d(:, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: tile_deltas_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        integer, pointer :: lbdd_CC_1_d_p(:)
        real, pointer :: scratch_hydro_op1_auxC_d_p(:, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_auxC_d_p(:)
        real, pointer :: scratch_hydro_op1_flZ_d_p(:, :, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_flZ_d_p(:)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        tile_deltas_d_p => tile_deltas_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        lbdd_CC_1_d_p => lbdd_CC_1_d(:, nblk)
        scratch_hydro_op1_auxC_d_p => scratch_hydro_op1_auxC_d(:, :, :, nblk)
        lbdd_scratch_hydro_op1_auxC_d_p => lbdd_scratch_hydro_op1_auxC_d(:, nblk)
        scratch_hydro_op1_flZ_d_p => scratch_hydro_op1_flZ_d(:, :, :, :, nblk)
        lbdd_scratch_hydro_op1_flZ_d_p => lbdd_scratch_hydro_op1_flZ_d(:, nblk)

        ! Call subroutine
        CALL Hydro_computeFluxesHll_Z_gpu_oacc( &
                    external_hydro_op1_dt_d, &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    tile_deltas_d_p, &
                    CC_1_d_p, &
                    lbdd_CC_1_d_p, &
                    scratch_hydro_op1_auxC_d_p, &
                    lbdd_scratch_hydro_op1_auxC_d_p, &
                    scratch_hydro_op1_flZ_d_p, &
                    lbdd_scratch_hydro_op1_flZ_d_p &
                )

    end subroutine wrapper_Hydro_computeFluxesHll_Z_gpu_oacc

    subroutine wrapper_Hydro_updateSolutionHll_gpu_oacc ( &
                    nblk, &
                    tile_lo_d, &
                    tile_hi_d, &
                    scratch_hydro_op1_flX_d, &
                    scratch_hydro_op1_flY_d, &
                    scratch_hydro_op1_flZ_d, &
                    lbdd_scratch_hydro_op1_flX_d, &
                    CC_1_d, &
                    lbdd_CC_1_d &
            )

        use dr_cg_hydroAdvance_mod, ONLY: Hydro_updateSolutionHll_gpu_oacc

        !$acc routine vector
        !$acc routine (Hydro_updateSolutionHll_gpu_oacc) vector

        implicit none

        ! Arguments
        integer, intent(IN) :: nblk
        integer, target, intent(IN) :: tile_lo_d(:, :)
        integer, target, intent(IN) :: tile_hi_d(:, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flX_d(:, :, :, :, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flY_d(:, :, :, :, :)
        real, target, intent(INOUT) :: scratch_hydro_op1_flZ_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_scratch_hydro_op1_flX_d(:, :)
        real, target, intent(INOUT) :: CC_1_d(:, :, :, :, :)
        integer, target, intent(IN) :: lbdd_CC_1_d(:, :)

        ! Local variables
        integer, pointer :: tile_lo_d_p(:)
        integer, pointer :: tile_hi_d_p(:)
        real, pointer :: scratch_hydro_op1_flX_d_p(:, :, :, :)
        real, pointer :: scratch_hydro_op1_flY_d_p(:, :, :, :)
        real, pointer :: scratch_hydro_op1_flZ_d_p(:, :, :, :)
        integer, pointer :: lbdd_scratch_hydro_op1_flX_d_p(:)
        real, pointer :: CC_1_d_p(:, :, :, :)
        integer, pointer :: lbdd_CC_1_d_p(:)

        ! Attach pointers
        tile_lo_d_p => tile_lo_d(:, nblk)
        tile_hi_d_p => tile_hi_d(:, nblk)
        scratch_hydro_op1_flX_d_p => scratch_hydro_op1_flX_d(:, :, :, :, nblk)
        scratch_hydro_op1_flY_d_p => scratch_hydro_op1_flY_d(:, :, :, :, nblk)
        scratch_hydro_op1_flZ_d_p => scratch_hydro_op1_flZ_d(:, :, :, :, nblk)
        lbdd_scratch_hydro_op1_flX_d_p => lbdd_scratch_hydro_op1_flX_d(:, nblk)
        CC_1_d_p => CC_1_d(:, :, :, :, nblk)
        lbdd_CC_1_d_p => lbdd_CC_1_d(:, nblk)

        ! Call subroutine
        CALL Hydro_updateSolutionHll_gpu_oacc( &
                    tile_lo_d_p, &
                    tile_hi_d_p, &
                    scratch_hydro_op1_flX_d_p, &
                    scratch_hydro_op1_flY_d_p, &
                    scratch_hydro_op1_flZ_d_p, &
                    lbdd_scratch_hydro_op1_flX_d_p, &
                    CC_1_d_p, &
                    lbdd_CC_1_d_p &
                )

    end subroutine wrapper_Hydro_updateSolutionHll_gpu_oacc

end module gpu_tf_hydro_mod

