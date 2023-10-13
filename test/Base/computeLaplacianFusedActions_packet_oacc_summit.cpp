#include "computeLaplacianFused.h"

#include <Milhoja.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_DataPacket.h>

#include "Base.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"

#include "DataPacket_gpu_tf_fused_actions.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit(const int tId,
                                                                     milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket_gpu_tf_fused_actions* packet_h   = dynamic_cast<DataPacket_gpu_tf_fused_actions*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);

    const std::size_t*  nTiles_d = packet_h->_nTiles_d;
    const RealVect* deltas_d = packet_h->_tile_deltas_d;
    const IntVect* lo_d = packet_h->_tile_lo_d;
    const IntVect* hi_d = packet_h->_tile_hi_d;
    const FArray4D* CC1_d = packet_h->_f4_Uin_d;
    FArray4D* CC2_d = packet_h->_f4_Uout_d;

    #pragma acc data deviceptr(nTiles_d, deltas_d, lo_d, hi_d, CC1_d, CC2_d)
    {
        // Wait for data to arrive and then
        // launch these two independent kernels for concurrent execution
        #pragma acc wait(queue_h)
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const FArray4D*        Uin_d  = CC1_d + n;
            FArray4D*              Uout_d = CC2_d + n;
            StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(lo_d + n, hi_d + n,
                                                                       Uin_d, Uout_d,
                                                                       deltas_d + n);
        }

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const FArray4D*        Uin_d  = CC1_d + n;
            FArray4D*              Uout_d = CC2_d + n;
            StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(lo_d + n, hi_d + n,
                                                                      Uin_d, Uout_d,
                                                                      deltas_d + n);
        }
    }
    #pragma acc wait(queue_h,queue2_h)

    packet_h->releaseExtraQueue(2);
}

