#ifndef USE_OPENACC
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "scaleEnergy.h"

#include "DataItem.h"
#include "DataPacket.h"

#include "Flash.h"

void ActionRoutines::scaleEnergy_packet_oacc_summit(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*   packet_h  = dynamic_cast<DataPacket*>(dataItem_h);
    const int     queue_h   = packet_h->asynchronousQueue();

    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);

    // TODO: This data should be included in the copyin section of the
    //       data packet.
    constexpr Real    ENERGY_SCALE_FACTOR = 5.0;

    // Computation done in-place 
    FArray4D*   U_d = nullptr;
    for (std::size_t n=0; n<packet_h->nTiles(); ++n) {
        const PacketContents  ptrs = packet_h->tilePointers(n);

        switch (packet_h->getDataLocation()) {
            case PacketDataLocation::CC1:
                U_d = ptrs.CC1_d;
                break;
            case PacketDataLocation::CC2:
                U_d = ptrs.CC2_d;
                break;
            default:
                throw std::logic_error("[scaleEnergy_packet_oacc_summit] "
                                       "Data not in CC1 or CC2");
        }

        StaticPhysicsRoutines::scaleEnergy_oacc_summit(ptrs.lo_d, ptrs.hi_d,
                                                       ptrs.xCoords_d, ptrs.yCoords_d,
                                                       U_d, ENERGY_SCALE_FACTOR,
                                                       queue_h);
    }
}

