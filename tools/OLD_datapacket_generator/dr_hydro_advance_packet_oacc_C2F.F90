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
C_dataQ_h, &
C_nTiles_h, &
C_nxbGC_h, &
C_nybGC_h, &
C_nzbGC_h, &
C_nCcVar_h, &
C_nFluxVar_h, &
C_nTiles_d, &
C_dt_start_d, &
C_deltas_start_d, &
C_lo_start_d, &
C_hi_start_d, &
C_U_start_d, &
C_auxC_start_d, &
C_FCX_start_d, &
C_FCY_start_d
) bind(c)
	use iso_c_binding, ONLY : C_PTR, & C_F_POINTER
	use openacc, ONLY : acc_handle_kind
	use milhoja_types_mod, ONLY : MILHOJA_INT
	use dr_hydroAdvance_bundle_mod, ONLY : dr_hydroAdvance_packet_gpu_oacc
	implicit none

	type(C_PTR), intent(IN), value :: C_packet_h
	integer(MILHOJA_INT), intent(IN), value :: C_dataQ_h
	integer(MILHOJA_INT), intent(IN), value :: C_nTiles_h
	integer(MILHOJA_INT), intent(IN), value :: C_nxbGC_h
	integer(MILHOJA_INT), intent(IN), value :: C_nybGC_h
	integer(MILHOJA_INT), intent(IN), value :: C_nzbGC_h
	integer(MILHOJA_INT), intent(IN), value :: C_nCcVar_h
	integer(MILHOJA_INT), intent(IN), value :: C_nFluxVar_h
	type(C_PTR), intent(IN), value :: C_nTiles_d
	type(C_PTR), intent(IN), value :: C_dt_start_d
	type(C_PTR), intent(IN), value :: C_deltas_start_d
	type(C_PTR), intent(IN), value :: C_lo_start_d
	type(C_PTR), intent(IN), value :: C_hi_start_d
	type(C_PTR), intent(IN), value :: C_U_start_d
	type(C_PTR), intent(IN), value :: C_auxC_start_d
	type(C_PTR), intent(IN), value :: C_FCX_start_d
	type(C_PTR), intent(IN), value :: C_FCY_start_d

	integer(kind=acc_handle_kind) :: F_dataQ_h
	integer :: F_nTiles_h
	integer :: F_nxbGC_h
	integer :: F_nybGC_h
	integer :: F_nzbGC_h
	integer :: F_nCcVar_h
	integer :: F_nFluxVar_h

	integer, pointer :: F_nTiles_d
	real, pointer :: F_dt_start_d
	real, pointer :: F_deltas_start_d(:,:)
	integer, pointer :: F_lo_start_d(:,:)
	integer, pointer :: F_hi_start_d(:,:)
	real, pointer :: F_U_start_d(:,:,:,:,:)
	real, pointer :: F_auxC_start_d(:,:,:,:,:)
	real, pointer :: F_FCX_start_d(:,:,:,:,:)
	real, pointer :: F_FCY_start_d(:,:,:,:,:)

	F_dataQ_h = INT(C_dataQ_h, kind=acc_handle_kind)
	F_nTiles_h = INT(C_nTiles_h)
	F_nxbGC_h = INT(C_nxbGC_h)
	F_nybGC_h = INT(C_nybGC_h)
	F_nzbGC_h = INT(C_nzbGC_h)
	F_nCcVar_h = INT(C_nCcVar_h)
	F_nFluxVar_h = INT(C_nFluxVar_h)

	CALL C_F_POINTER(C_dataQ_h, F_dataQ_h)
	CALL C_F_POINTER(C_nTiles_h, F_nTiles_h)
	CALL C_F_POINTER(C_nxbGC_h, F_nxbGC_h)
	CALL C_F_POINTER(C_nybGC_h, F_nybGC_h)
	CALL C_F_POINTER(C_nzbGC_h, F_nzbGC_h)
	CALL C_F_POINTER(C_nCcVar_h, F_nCcVar_h)
	CALL C_F_POINTER(C_nFluxVar_h, F_nFluxVar_h)
	CALL C_F_POINTER(C_nTiles_d, F_nTiles_d)
	CALL C_F_POINTER(C_dt_start_d, F_dt_start_d)
	CALL C_F_POINTER(C_deltas_start_d, F_deltas_start_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_lo_start_d, F_lo_start_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_hi_start_d, F_hi_start_d, shape=[3, F_nTiles_h])
	CALL C_F_POINTER(C_U_start_d, F_U_start_d, shape=[F_nxbGC_h, F_nybGC_h, F_nzbGC_h, ( (GAMC_VAR) + (UNK_VARS_BEGIN) + 1 ), F_nTiles_h])
	CALL C_F_POINTER(C_auxC_start_d, F_auxC_start_d, shape=[F_nxbGC_h, F_nybGC_h, F_nzbGC_h, ( (0) + (0) + 1 ), F_nTiles_h])
	CALL C_F_POINTER(C_FCX_start_d, F_FCX_start_d, shape=[F_nxbGC_h + 1, F_nybGC_h, F_nzbGC_h, ( (4) + (0) + 1 ), F_nTiles_h])
	CALL C_F_POINTER(C_FCY_start_d, F_FCY_start_d, shape=[F_nxbGC_h, F_nybGC_h + 1, F_nzbGC_h, ( (4) + (0) + 1 ), F_nTiles_h])

	CALL dr_hydroAdvance_packet_gpu_oacc(C_packet_h, &
		F_dataQ_h, &
		F_nTiles_h, &
		F_nxbGC_h, &
		F_nybGC_h, &
		F_nzbGC_h, &
		F_nCcVar_h, &
		F_nFluxVar_h, &
		F_nTiles_d, &
		F_dt_start_d, &
		F_deltas_start_d, &
		F_lo_start_d, &
		F_hi_start_d, &
		F_U_start_d, &
		F_auxC_start_d, &
		F_FCX_start_d, &
		F_FCY_start_d)
end subroutine dr_hydro_advance_packet_oacc_c2f