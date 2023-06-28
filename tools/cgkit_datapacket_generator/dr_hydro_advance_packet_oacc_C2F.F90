!! This code was generated using C2F_generator.py.
!> @copyright Copyright 2022 UChicago Argonne, LLC and contributors
!!
!! @licenseblock
!! Licensed under the Apache License, Version 2.0 (the "License");
!! you may not use this file except in compliance with the License.
!!
!! Unless required by applicable law or agreed to in writing, software
!! distributed under the License is distributed on an "AS IS" BASIS,
!! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!! See the License for the specific language governing permissions and
!! limitations under the License.
!! @endlicenseblock
!!
!! @file

#include "Milhoja.h"
#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

subroutine dr_hydro_advance_packet_oacc_c2f(C_packet_h, &
C_queue1_h, &
C_nTiles_h, &
C_nTiles_d, &
C_dt_d, &
C_deltas_d, &
C_lo_d, &
C_hi_d, &
C_U_d, &
C_auxC_d, &
C_flX_d, &
C_flY_d, &
C_flZ_d) bind(c)
	use iso_c_binding, ONLY : C_PTR, C_F_POINTER
	use openacc, ONLY : acc_handle_kind
	use milhoja_types_mod, ONLY : MILHOJA_INT
	use dr_hydroAdvance_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc
	implicit none

	type(C_PTR), intent(IN), value :: C_packet_h
	integer(MILHOJA_INT), intent(IN), value :: C_queue1_h
	integer(MILHOJA_INT), intent(IN), value :: C_nTiles_h

	type(C_PTR), intent(IN), value :: C_nTiles_d
	type(C_PTR), intent(IN), value :: C_dt_d
	type(C_PTR), intent(IN), value :: C_deltas_d
	type(C_PTR), intent(IN), value :: C_lo_d
	type(C_PTR), intent(IN), value :: C_hi_d
	type(C_PTR), intent(IN), value :: C_U_d
	type(C_PTR), intent(IN), value :: C_auxC_d
	type(C_PTR), intent(IN), value :: C_flX_d
	type(C_PTR), intent(IN), value :: C_flY_d
	type(C_PTR), intent(IN), value :: C_flZ_d

	integer(kind=acc_handle_kind) :: F_queue1_h
	integer :: F_nTiles_h

	int, pointer :: F_nTiles_d
	real, pointer :: F_dt_d
	real, pointer :: F_deltas_d(:,:)
	integer, pointer :: F_lo_d(:,:)
	integer, pointer :: F_hi_d(:,:)
	real, pointer :: F_U_d(:,:,:,:,:)
	real, pointer :: F_auxC_d(:,:,:,:)
	real, pointer :: F_flX_d(:,:,:,:,:)
	real, pointer :: F_flY_d(:,:,:,:,:)
	real, pointer :: F_flZ_d(:,:,:,:,:)

	F_packet_h = INT(C_packet_h)
	F_queue1_h = INT(C_queue1_h, kind=acc_handle_kind)
	F_nTiles_h = INT(C_nTiles_h)

	CALL C_F_POINTER(C_nTiles_d, F_nTiles_d)
	CALL C_F_POINTER(C_dt_d, F_dt_d)
	CALL C_F_POINTER(C_deltas_d, F_deltas_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_lo_d, F_lo_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_hi_d, F_hi_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_U_d, F_U_d, shape=[8 + 2 * 1, 8 + 2 * 1, 1 + 2 * 0, 8 - 0 + 1, F_nTiles_h])
	CALL C_F_POINTER(C_auxC_d, F_auxC_d, shape=[8 + 2 * 1, 8 + 2 * 1, 1 + 2 * 0, F_nTiles_h])
	CALL C_F_POINTER(C_flX_d, F_flX_d, shape=[(8 + 2 * 1) + 1, 8 + 2 * 1, 1 + 2 * 0, 5, F_nTiles_h])
	CALL C_F_POINTER(C_flY_d, F_flY_d, shape=[8 + 2 * 1, (8 + 2 * 1) + 1, 1 + 2 * 0, 5, F_nTiles_h])
	CALL C_F_POINTER(C_flZ_d, F_flZ_d, shape=[1, 1, 1, 1, F_nTiles_h])

	CALL dr_hydroAdvance_packet_gpu_oacc(C_packet_h, &
		F_queue1_h, &
		C_nTiles_h, &
		C_dt_h, &
		C_lo_h, &
		C_hi_h, &
		C_deltas_h, &
		C_loU_h, &
		C_U_h, &
		C_loAuxC_h, &
		C_auxC_h, &
		C_loFl_h, &
		C_flX_h, &
		C_flY_h, &
		C_flZ_h)
end subroutine dr_hydro_advance_packet_oacc_c2f