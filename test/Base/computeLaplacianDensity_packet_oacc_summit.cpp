#ifndef USE_OPENACC
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianDensity.h"

#include "DataItem.h"
#include "DataPacket.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianDensity_packet_oacc_summit(const int tId,
                                                                orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*   packet_h  = dynamic_cast<DataPacket*>(dataItem_h);
    const int     queue_h   = packet_h->asynchronousQueue();

    packet_h->setVariableMask(DENS_VAR_C, DENS_VAR_C);

    // Data will be written to Uout
    FArray4D*   Uin_d  = nullptr;
    FArray4D*   Uout_d = nullptr;
    for (std::size_t n=0; n<packet_h->nTiles(); ++n) {
        const PacketContents&  ptrs = packet_h->tilePointers(n);

        switch (packet_h->getDataLocation()) {
            case PacketDataLocation::CC1:
                Uin_d  = ptrs.CC1_d;
                Uout_d = ptrs.CC2_d;
                packet_h->setDataLocation(PacketDataLocation::CC2);
                break;
            case PacketDataLocation::CC2:
                Uin_d  = ptrs.CC2_d;
                Uout_d = ptrs.CC1_d;
                packet_h->setDataLocation(PacketDataLocation::CC1);
                break;
            default:
                throw std::logic_error("[computeLaplacianDensity_packet_oacc_summit] "
                                       "Data not in CC1 or CC2");
        }

        StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(ptrs.lo_d, ptrs.hi_d,
                                                                   Uin_d, Uout_d,
                                                                   ptrs.deltas_d,
                                                                   queue_h);
    }
}

