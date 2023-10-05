

#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>

#include "DataPacket_Hydro_gpu_3.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

using milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void dr_hydroAdvance_packet_gpu_oacc_c2f (
    void* packet_h,
    const int queue1_h,
    const int _nTiles_h,
    const void* _nTiles_d,
    const void* _dt_d,
    const void* _tile_deltas_d,
    const void* _tile_lo_d,
    const void* _tile_hi_d,
    const void* _tile_loGC_d,
    const void* _U_d,
    const void* _auxC_d,
    const void* _FCX_d,
    const void* _FCY_d,
    const void* _FCZ_d
    
    );

    int instantiate_DataPacket_Hydro_gpu_3_C (
    real dt,void** packet
    
        ) {
        if ( packet == nullptr) {
            std::cerr << "[instantiate_DataPacket_Hydro_gpu_3_C] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*packet != nullptr) {
            std::cerr << "[instantiate_DataPacket_Hydro_gpu_3_C] *packet not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *packet = static_cast<void*>(new DataPacket_Hydro_gpu_3(
            dt
            ));
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        } catch (...) {
            std::cerr << "[instantiate_DataPacket_Hydro_gpu_3_C] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        }

        return MILHOJA_SUCCESS;
    }

    int delete_DataPacket_Hydro_gpu_3_C (void* packet) {
        if (packet == nullptr) {
            std::cerr << "[delete_DataPacket_Hydro_gpu_3_C] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast< DataPacket_Hydro_gpu_3 *>(packet);
        return MILHOJA_SUCCESS;
    }

    int release_DataPacket_Hydro_gpu_3_extra_queue_C (void* packet, const int id) {
        std::cerr << "[release_DataPacket_Hydro_gpu_3_extra_queue_C] Packet does not have extra queues." << std::endl;
        return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
    }

    //----- C TASK FUNCTION TO BE CALLED BY RUNTIME
    void dr_hydroAdvance_packet_gpu_oacc_tf (const int tId, void* dataItem_h) {
        DataPacket_Hydro_gpu_3* packet_h = static_cast<DataPacket_Hydro_gpu_3*>(dataItem_h);
        const int queue1_h = packet_h->asynchronousQueue();
        const int _nTiles_h = packet_h->_nTiles_h;
        

        void* _nTiles_d = static_cast<void*>( packet_h->_nTiles_d );
        void* _dt_d = static_cast<void*>( packet_h->_dt_d );
        void* _tile_deltas_d = static_cast<void*>( packet_h->_tile_deltas_d );
        void* _tile_lo_d = static_cast<void*>( packet_h->_tile_lo_d );
        void* _tile_hi_d = static_cast<void*>( packet_h->_tile_hi_d );
        void* _tile_loGC_d = static_cast<void*>( packet_h->_tile_loGC_d );
        void* _U_d = static_cast<void*>( packet_h->_U_d );
        void* _auxC_d = static_cast<void*>( packet_h->_auxC_d );
        void* _FCX_d = static_cast<void*>( packet_h->_FCX_d );
        void* _FCY_d = static_cast<void*>( packet_h->_FCY_d );
        void* _FCZ_d = static_cast<void*>( packet_h->_FCZ_d );
        

        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        dr_hydroAdvance_packet_gpu_oacc_c2f (
        packet_h,
        queue1_h,
        _nTiles_h,
        _nTiles_d,
        _dt_d,
        _tile_deltas_d,
        _tile_lo_d,
        _tile_hi_d,
        _tile_loGC_d,
        _U_d,
        _auxC_d,
        _FCX_d,
        _FCY_d,
        _FCZ_d
        
        );
    }
}