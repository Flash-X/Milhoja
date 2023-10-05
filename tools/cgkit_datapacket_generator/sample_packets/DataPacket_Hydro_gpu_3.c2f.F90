!! This code was generated using c2f_generator.py.
#include "Milhoja.h"
#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

subroutine dr_hydroAdvance_packet_gpu_oacc_c2f(C_packet_h, &
C_queue1_h, &
C_nTiles_h, &
C_nTiles_d, &
C_dt_d, &
C_tile_deltas_d, &
C_tile_lo_d, &
C_tile_hi_d, &
C_tile_loGC_d, &
C_U_d, &
C_auxC_d, &
C_FCX_d, &
C_FCY_d, &
C_FCZ_d) bind(c)
	use iso_c_binding, ONLY : C_PTR, C_F_POINTER
	use openacc, ONLY : acc_handle_kind
	use milhoja_types_mod, ONLY : MILHOJA_INT
	use dr_hydroAdvance_packet_gpu_oacc_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc
	implicit none

	type(C_PTR), intent(IN), value :: C_packet_h
	integer(MILHOJA_INT), intent(IN), value :: C_queue1_h
	integer(MILHOJA_INT), intent(IN), value :: C_nTiles_h

	type(C_PTR), intent(IN), value :: C_nTiles_d
	type(C_PTR), intent(IN), value :: C_dt_d
	type(C_PTR), intent(IN), value :: C_tile_deltas_d
	type(C_PTR), intent(IN), value :: C_tile_lo_d
	type(C_PTR), intent(IN), value :: C_tile_hi_d
	type(C_PTR), intent(IN), value :: C_tile_loGC_d
	type(C_PTR), intent(IN), value :: C_U_d
	type(C_PTR), intent(IN), value :: C_auxC_d
	type(C_PTR), intent(IN), value :: C_FCX_d
	type(C_PTR), intent(IN), value :: C_FCY_d
	type(C_PTR), intent(IN), value :: C_FCZ_d

	integer(kind=acc_handle_kind) :: F_queue1_h
	integer :: F_nTiles_h

	integer, pointer :: F_nTiles_d
	real, pointer :: F_dt_d
	real, pointer :: F_tile_deltas_d(:,:)
	integer, pointer :: F_tile_lo_d(:,:)
	integer, pointer :: F_tile_hi_d(:,:)
	integer, pointer :: F_tile_loGC_d(:,:)
	real, pointer :: F_U_d(:,:,:,:,:)
	real, pointer :: F_auxC_d(:,:,:,:)
	real, pointer :: F_FCX_d(:,:,:,:,:)
	real, pointer :: F_FCY_d(:,:,:,:,:)
	real, pointer :: F_FCZ_d(:,:,:,:,:)

	F_queue1_h = INT(C_queue1_h, kind=acc_handle_kind)
	F_nTiles_h = INT(C_nTiles_h)

	CALL C_F_POINTER(C_nTiles_d, F_nTiles_d)
	CALL C_F_POINTER(C_dt_d, F_dt_d)
	CALL C_F_POINTER(C_tile_deltas_d, F_tile_deltas_d, shape=[MILHOJA_MDIM, F_nTiles_h])
	CALL C_F_POINTER(C_tile_lo_d, F_tile_lo_d, shape=[MILHOJA_MDIM, F_nTiles_h])
	CALL C_F_POINTER(C_tile_hi_d, F_tile_hi_d, shape=[MILHOJA_MDIM, F_nTiles_h])
	CALL C_F_POINTER(C_tile_loGC_d, F_tile_loGC_d, shape=[MILHOJA_MDIM, F_nTiles_h])
	CALL C_F_POINTER(C_U_d, F_U_d, shape=[8 + 2 * 1, 8 + 2 * 1, 1 + 2 * 0, 9 - 0 + 1, F_nTiles_h])
	CALL C_F_POINTER(C_auxC_d, F_auxC_d, shape=[8 + 2 * 1, 8 + 2 * 1, 1 + 2 * 0, F_nTiles_h])
	CALL C_F_POINTER(C_FCX_d, F_FCX_d, shape=[(8 + 2 * 1) + 1, 8 + 2 * 1, 1 + 2 * 0, 5, F_nTiles_h])
	CALL C_F_POINTER(C_FCY_d, F_FCY_d, shape=[8 + 2 * 1, (8 + 2 * 1) + 1, 1 + 2 * 0, 5, F_nTiles_h])
	CALL C_F_POINTER(C_FCZ_d, F_FCZ_d, shape=[1, 1, 1, 1, F_nTiles_h])

	CALL dr_hydroAdvance_packet_gpu_oacc(C_packet_h, &
		F_queue1_h, &
		F_nTiles_d, &
		F_dt_d, &
		F_tile_deltas_d, &
		F_tile_lo_d, &
		F_tile_hi_d, &
		F_tile_loGC_d, &
		F_U_d, &
		F_auxC_d, &
		F_FCX_d, &
		F_FCY_d, &
		F_FCZ_d)
end subroutine dr_hydroAdvance_packet_gpu_oacc_c2f