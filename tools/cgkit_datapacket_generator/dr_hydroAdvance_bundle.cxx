
/**
 * @copyright Copyright 2022 UChicago Argonne, LLC and contributors
 *
 * @licenseblock
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * @endlicenseblock
 *
 * @file
 */
 #include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>
#include "cgkit_dataPacket_Hydro_gpu_3_2nd_pass.h"
#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif
extern "C" {
	// C DECLARATION OF FORTRAN INTERFACE
	void dr_hydro_advance_packet_oacc_c2f(
		void* packet_h
		const int queue1_h
		const int nTiles_h
		const void* nTiles_d,
		const void* dt_d,
		const void* nTiles_d,
		const void* tile_deltas_d,
		const void* tile_lo_d,
		const void* tile_hi_d,
		const void* U_d,
		const void* auxC_d,
		const void* flX_d,
		const void* flY_d,
		const void* flZ_d)
