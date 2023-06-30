

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

// include the datapacket. this may need to change depending on how the name is determined. for now,
// cgkit outputs will be named cgkit.datapacket.h.
#include "cgkit.datapacket.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

using milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void dr_hydro_advance_packet_oacc_c2f(
  void* packet_h,
  const int queue1_h,
  const void* _dt_d,
  const void* _lo_d,
  const void* _hi_d,
  const void* _deltas_d,
  const void* _loU_d,
  const void* _U_d,
  const void* _loAuxC_d,
  const void* _auxC_d,
  const void* _loFl_d,
  const void* _flX_d,
  const void* _flY_d,
  const void* _flZ_d
  
        // void* packet_h,
        // const int dataQ_h,
        // const int queue2_h,
        // const int queue3_h,
        // const int nTiles_h,
        // const void* nTiles_d, const void* dt_d,
        // const void* deltas_start_d,
        // const void* lo_start_d,   const void* hi_start_d,
        // const void* loGC_start_d,
        // const void* U_start_d,
        // const void* auxC_start_d,
        // const void* faceX_start_d,
        // const void* faceY_start_d,
        // const void* faceZ_start_d
    );

    int instantiate_hydro_advance_packet_c(
  const Real dt
  
        ,void** packet) {
        if ( packet == nullptr) {
            std::cerr << "[instantiate_hydro_advance_packet_c] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*packet != nullptr) {
            std::cerr << "[instantiate_hydro_advance_packet_c] *packet not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *packet = static_cast<void*>(new cgkit_dataPacket_Hydro_gpu_3_2nd_pass(
      dt
                // dt
            ));
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        } catch (...) {
            std::cerr << "[instantiate_hydro_advance_packet_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        }

        return MILHOJA_SUCCESS;
    }

    int delete_hydro_advance_packet_c(void* packet) {
        if (packet == nullptr) {
            std::cerr << "[delete_hydro_advance_packet_c] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast< cgkit_dataPacket_Hydro_gpu_3_2nd_pass *>(packet);
        return MILHOJA_SUCCESS;
    }

    int release_hydro_advance_extra_queue_c(void* packet, const int id) {
        if (packet == nullptr) {
            std::cerr << "[release_hydro_advance_extra_queue_c] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        cgkit_dataPacket_Hydro_gpu_3_2nd_pass*   packet_h = static_cast<cgkit_dataPacket_Hydro_gpu_3_2nd_pass*>(packet);

        try {
            _param:releaseQueue
            // packet_h->releaseExtraQueue(id);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
        } catch (...) {
            std::cerr << "[release_hydro_advance_extra_queue_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
        }

        return MILHOJA_SUCCESS;
    }

    //----- C TASK FUNCTION TO BE CALLED BY RUNTIME
    void dr_hydro_advance_packet_oacc_tf(const int tId, void* dataItem_h) {
        cgkit_dataPacket_Hydro_gpu_3_2nd_pass* packet_h = static_cast<cgkit_dataPacket_Hydro_gpu_3_2nd_pass *>(dataItem_h);
        const milhoja::PacketDataLocation   location = packet_h->getDataLocation();
    const int queue1_h = packet_h->asynchronousQueue();
    const int nTiles_h = packet_h->_nTiles_h;
    
        // const int                           dataQ_h  = packet_h->asynchronousQueue();
        // const int                           queue2_h = packet_h->extraAsynchronousQueue(2);
        // const int                           queue3_h = packet_h->extraAsynchronousQueue(3);
        // const int                           nTiles_h = packet_h->nTiles_host();

    void* _dt_d = static_cast<void*>( packet_h->_dt_d );
    void* _lo_d = static_cast<void*>( packet_h->_lo_d );
    void* _hi_d = static_cast<void*>( packet_h->_hi_d );
    void* _deltas_d = static_cast<void*>( packet_h->_deltas_d );
    void* _loU_d = static_cast<void*>( packet_h->_loU_d );
    void* _U_d = static_cast<void*>( packet_h->_U_d );
    void* _loAuxC_d = static_cast<void*>( packet_h->_loAuxC_d );
    void* _auxC_d = static_cast<void*>( packet_h->_auxC_d );
    void* _loFl_d = static_cast<void*>( packet_h->_loFl_d );
    void* _flX_d = static_cast<void*>( packet_h->_flX_d );
    void* _flY_d = static_cast<void*>( packet_h->_flY_d );
    void* _flZ_d = static_cast<void*>( packet_h->_flZ_d );
    
        // int*              nTiles_d       = packet_h->nTiles_devptr();
        // milhoja::Real*    dt_d           = packet_h->dt_devptr();
        // milhoja::Real*    deltas_start_d = packet_h->deltas_devptr();
        // int*              lo_start_d     = packet_h->lo_devptr();
        // int*              hi_start_d     = packet_h->hi_devptr();
        // int*              loGC_start_d   = packet_h->loGC_devptr();
        // milhoja::Real*    U_start_d      = packet_h->U_devptr();
        // milhoja::Real*    auxC_start_d   = packet_h->scratchAuxC_devptr();
        // milhoja::Real*    faceX_start_d  = packet_h->scratchFaceX_devptr();
        // milhoja::Real*    faceY_start_d  = packet_h->scratchFaceY_devptr();
        // milhoja::Real*    faceZ_start_d  = packet_h->scratchFaceZ_devptr();

        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        dr_hydro_advance_packet_oacc_c2f(
    packet_h,
    queue1_h,
    _dt_d,
    _lo_d,
    _hi_d,
    _deltas_d,
    _loU_d,
    _U_d,
    _loAuxC_d,
    _auxC_d,
    _loFl_d,
    _flX_d,
    _flY_d,
    _flZ_d
    
            // packet_h,
            // dataQ_h, queue2_h, queue3_h,
            // nTiles_h,
            // nTiles_d, dt_d,
            // deltas_start_d,
            // lo_start_d,   hi_start_d,
            // loGC_start_d,
            // U_start_d,
            // auxC_start_d,
            // faceX_start_d, faceY_start_d, faceZ_start_d
        );
    }
}