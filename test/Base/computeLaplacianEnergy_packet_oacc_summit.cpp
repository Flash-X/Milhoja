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
    
    DataPacket*                packet_h = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h  = packet_h->asynchronousQueue();
    const PacketDataLocation   location = packet_h->getDataLocation();

    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);
    switch (location) {
        case PacketDataLocation::CC1:
            packet_h->setDataLocation(PacketDataLocation::CC2);
            break;
        case PacketDataLocation::CC2:
            packet_h->setDataLocation(PacketDataLocation::CC1);
            break;
        default:
            throw std::logic_error("[computeLaplacianEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }

    // Data will be written to Uout
    FArray4D*   Uin_d  = nullptr;
    FArray4D*   Uout_d = nullptr;
    for (std::size_t n=0; n<packet_h->nTiles(); ++n) {
        const PacketContents&  ptrs = packet_h->tilePointers(n);

        switch (location) {
            case PacketDataLocation::CC1:
                Uin_d  = ptrs.CC1_d;
                Uout_d = ptrs.CC2_d;
                break;
            case PacketDataLocation::CC2:
                Uin_d  = ptrs.CC2_d;
                Uout_d = ptrs.CC1_d;
                break;
            default:
                throw std::logic_error("[computeLaplacianEnergy_packet_oacc_summit] "
                                       "Data not in CC1 or CC2");
        }

        StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs.lo_d, ptrs.hi_d,
                                                                  Uin_d, Uout_d,
                                                                  ptrs.deltas_d,
                                                                  queue_h);
    }
}

