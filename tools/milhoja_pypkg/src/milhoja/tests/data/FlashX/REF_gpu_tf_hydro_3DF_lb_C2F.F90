#include "Milhoja.h"
#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

subroutine gpu_tf_hydro_C2F( &
C_packet_h, &
C_queue1_h, &
C_queue2_h, &
C_queue3_h, &
C_nTiles_h, &
C_nTiles_d, &
C_external_hydro_op1_dt_d, &
C_tile_deltas_d, &
C_tile_lo_d, &
C_tile_hi_d, &
C_tile_interior_d, &
C_tile_lbound_d, &
C_tile_ubound_d, &
C_tile_arrayBounds_d, &
C_lbdd_CC_1_d, &
C_CC_1_d, &
C_lbdd_scratch_hydro_op1_auxC_d, &
C_scratch_hydro_op1_auxC_d, &
C_scratch_hydro_op1_flX_d, &
C_scratch_hydro_op1_flY_d, &
C_scratch_hydro_op1_flZ_d &
) bind(c, name="gpu_tf_hydro_C2F")
    use iso_c_binding, ONLY : C_PTR, C_F_POINTER
    use openacc, ONLY : acc_handle_kind
    use milhoja_types_mod, ONLY : MILHOJA_INT
    use gpu_tf_hydro_mod, ONLY : gpu_tf_hydro_Fortran
    implicit none

    type(C_PTR), intent(IN), value :: C_packet_h
    integer(MILHOJA_INT), intent(IN), value :: C_queue1_h
    integer(MILHOJA_INT), intent(IN), value :: C_queue2_h
    integer(MILHOJA_INT), intent(IN), value :: C_queue3_h
    integer(MILHOJA_INT), intent(IN), value :: C_nTiles_h

    type(C_PTR), intent(IN), value :: C_nTiles_d
    type(C_PTR), intent(IN), value :: C_external_hydro_op1_dt_d
    type(C_PTR), intent(IN), value :: C_tile_deltas_d
    type(C_PTR), intent(IN), value :: C_tile_lo_d
    type(C_PTR), intent(IN), value :: C_tile_hi_d
    type(C_PTR), intent(IN), value :: C_tile_interior_d
    type(C_PTR), intent(IN), value :: C_tile_lbound_d
    type(C_PTR), intent(IN), value :: C_tile_ubound_d
    type(C_PTR), intent(IN), value :: C_tile_arrayBounds_d
    type(C_PTR), intent(IN), value :: C_lbdd_CC_1_d
    type(C_PTR), intent(IN), value :: C_CC_1_d
    type(C_PTR), intent(IN), value :: C_lbdd_scratch_hydro_op1_auxC_d
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_auxC_d
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flX_d
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flY_d
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flZ_d

    integer(kind=acc_handle_kind) :: F_queue1_h
    integer(kind=acc_handle_kind) :: F_queue2_h
    integer(kind=acc_handle_kind) :: F_queue3_h
    integer :: F_nTiles_h

    integer, pointer :: F_nTiles_d
    real, pointer :: F_external_hydro_op1_dt_d
    real, pointer :: F_tile_deltas_d(:,:)
    integer, pointer :: F_tile_lo_d(:,:)
    integer, pointer :: F_tile_hi_d(:,:)
    integer, pointer :: F_tile_interior_d(:,:,:)
    integer, pointer :: F_tile_lbound_d(:,:)
    integer, pointer :: F_tile_ubound_d(:,:)
    integer, pointer :: F_tile_arrayBounds_d(:,:,:)
    integer, pointer :: F_lbdd_CC_1_d(:,:)
    real, pointer :: F_CC_1_d(:,:,:,:,:)
    integer, pointer :: F_lbdd_scratch_hydro_op1_auxC_d(:,:)
    real, pointer :: F_scratch_hydro_op1_auxC_d(:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_flX_d(:,:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_flY_d(:,:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_flZ_d(:,:,:,:,:)

    F_queue1_h = INT(C_queue1_h, kind=acc_handle_kind)
    F_queue2_h = INT(C_queue2_h, kind=acc_handle_kind)
    F_queue3_h = INT(C_queue3_h, kind=acc_handle_kind)
    F_nTiles_h = INT(C_nTiles_h)

    CALL C_F_POINTER(C_nTiles_d, F_nTiles_d)
    CALL C_F_POINTER(C_external_hydro_op1_dt_d, F_external_hydro_op1_dt_d)
    CALL C_F_POINTER(C_tile_deltas_d, F_tile_deltas_d, shape=[MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_lo_d, F_tile_lo_d, shape=[MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_hi_d, F_tile_hi_d, shape=[MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_interior_d, F_tile_interior_d, shape=[2, MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_lbound_d, F_tile_lbound_d, shape=[MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_ubound_d, F_tile_ubound_d, shape=[MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_tile_arrayBounds_d, F_tile_arrayBounds_d, shape=[2, MILHOJA_MDIM, F_nTiles_h])
    CALL C_F_POINTER(C_lbdd_CC_1_d, F_lbdd_CC_1_d, shape=[4,F_nTiles_h])
    CALL C_F_POINTER(C_CC_1_d, F_CC_1_d, shape=[16 + 2 * 1 * MILHOJA_K1D, 16 + 2 * 1 * MILHOJA_K2D, 16 + 2 * 1 * MILHOJA_K3D, 9, F_nTiles_h])
    CALL C_F_POINTER(C_lbdd_scratch_hydro_op1_auxC_d, F_lbdd_scratch_hydro_op1_auxC_d, shape=[3, F_nTiles_h])
    CALL C_F_POINTER(C_scratch_hydro_op1_auxC_d, F_scratch_hydro_op1_auxC_d, shape=[18, 18, 18, F_nTiles_h])
    CALL C_F_POINTER(C_scratch_hydro_op1_flX_d, F_scratch_hydro_op1_flX_d, shape=[19, 18, 18, 5, F_nTiles_h])
    CALL C_F_POINTER(C_scratch_hydro_op1_flY_d, F_scratch_hydro_op1_flY_d, shape=[18, 19, 18, 5, F_nTiles_h])
    CALL C_F_POINTER(C_scratch_hydro_op1_flZ_d, F_scratch_hydro_op1_flZ_d, shape=[18, 18, 19, 5, F_nTiles_h])

    CALL gpu_tf_hydro_Fortran( &
        C_packet_h, &
        F_queue1_h, &
        F_queue2_h, &
        F_queue3_h, &
        F_nTiles_d, &
        F_external_hydro_op1_dt_d, &
        F_tile_deltas_d, &
        F_tile_lo_d, &
        F_tile_hi_d, &
        F_tile_interior_d, &
        F_tile_lbound_d, &
        F_tile_ubound_d, &
        F_tile_arrayBounds_d, &
        F_lbdd_CC_1_d, &
        F_CC_1_d, &
        F_lbdd_scratch_hydro_op1_auxC_d, &
        F_scratch_hydro_op1_auxC_d, &
        F_scratch_hydro_op1_flX_d, &
        F_scratch_hydro_op1_flY_d, &
        F_scratch_hydro_op1_flZ_d &
        )
end subroutine gpu_tf_hydro_C2F