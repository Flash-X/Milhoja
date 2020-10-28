#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianEnergy.h"

#include "DataItem.h"
#include "DataPacket.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianEnergy_packet_oacc_summit(const int tId,
                                                               orchestration::DataItem* dataItem_h) {
    using namespace orchestration;
    
    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const std::size_t*         nTiles_d   = packet_h->nTilesGpu();
    const PacketContents*      contents_d = packet_h->tilePointers();

    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        // Data will be written to Uout
        if        (location == PacketDataLocation::CC1) {
            packet_h->setDataLocation(PacketDataLocation::CC2);

            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        Uin_d  = ptrs->CC1_d;
                FArray4D*              Uout_d = ptrs->CC2_d;
                StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                          Uin_d, Uout_d,
                                                                          ptrs->deltas_d);
            }
        } else if (location == PacketDataLocation::CC2) {
            packet_h->setDataLocation(PacketDataLocation::CC1);

            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        Uin_d  = ptrs->CC2_d;
                FArray4D*              Uout_d = ptrs->CC1_d;
                StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                          Uin_d, Uout_d,
                                                                          ptrs->deltas_d);
            }
        } else {
            throw std::logic_error("[computeLaplacianEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
        }
    }

    #pragma acc wait(queue_h)
}

