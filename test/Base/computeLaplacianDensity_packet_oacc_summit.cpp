#include "computeLaplacianDensity.h"

#include <Milhoja.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_DataPacket.h>
#include "DataPacket_gpu_dens_stream.h"

#include "Base.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void ActionRoutines::computeLaplacianDensity_packet_oacc_summit(const int tId,
                                                                milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket_gpu_tf_dens* packet_h   = dynamic_cast<DataPacket_gpu_tf_dens*>(dataItem_h);
    const int                   queue_h    = packet_h->asynchronousQueue();

    const std::size_t* nTiles_d = packet_h->_nTiles_d;
    const RealVect* deltas_d = packet_h->_tile_deltas_d;
    const IntVect* lo_d = packet_h->_tile_lo_d;
    const IntVect* hi_d = packet_h->_tile_hi_d;
    FArray4D* CC1_d = packet_h->_f4_CC1_d;
    FArray4D* CC2_d = packet_h->_f4_CC2_d;

    #pragma acc data deviceptr(nTiles_d, deltas_d, lo_d, hi_d, CC1_d, CC2_d)
    {
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const FArray4D*        Uin_d  = CC1_d + n;
            FArray4D*              Uout_d = CC2_d + n;
	        const RealVect* deltas = deltas_d + n;
	        const IntVect* lo = lo_d + n;
	        const IntVect* hi = hi_d + n;
            StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(lo, hi,
                                                                       Uin_d, Uout_d,
                                                                       deltas);
        }
    }
    #pragma acc wait(queue_h)
}

