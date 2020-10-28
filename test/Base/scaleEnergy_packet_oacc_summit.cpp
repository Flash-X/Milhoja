#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "scaleEnergy.h"

#include "DataItem.h"
#include "DataPacket.h"

#include "Flash.h"

void ActionRoutines::scaleEnergy_packet_oacc_summit(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const std::size_t*         nTiles_d   = packet_h->nTilesGpu();
    const PacketContents*      contents_d = packet_h->tilePointers();

    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);

    // TODO: This data should be included in the copyin section of the
    //       data packet.
    constexpr Real    ENERGY_SCALE_FACTOR = 5.0;

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        // Computation done in-place 
        if        (location == PacketDataLocation::CC1) {
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*   ptrs = contents_d + n;
                FArray4D*               U_d = ptrs->CC1_d;
                StaticPhysicsRoutines::scaleEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                               ptrs->xCoords_d, ptrs->yCoords_d,
                                                               U_d, ENERGY_SCALE_FACTOR);
            }
        } else if (location == PacketDataLocation::CC2) {
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*   ptrs = contents_d + n;
                FArray4D*               U_d = ptrs->CC2_d;
                StaticPhysicsRoutines::scaleEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                               ptrs->xCoords_d, ptrs->yCoords_d,
                                                               U_d, ENERGY_SCALE_FACTOR);
            }
        } else {
            throw std::logic_error("[scaleEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
        }
    }

    #pragma acc wait(queue_h)
}

