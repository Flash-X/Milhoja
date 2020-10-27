#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianFused.h"

#include "DataItem.h"
#include "DataPacket.h"

#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianFusedKernelsWeak_packet_oacc_summit(const int tId,
                                                                         orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const std::size_t*         nTiles_d   = packet_h->nTilesGpu();
    const PacketContents*      contents_d = packet_h->tilePointers();

    packet_h->setVariableMask(DENS_VAR_C, ENER_VAR_C);

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        // Data will be written to Uout
        FArray4D*   Uin_d  = nullptr;
        FArray4D*   Uout_d = nullptr;
        switch (location) {
            case PacketDataLocation::CC1:
                packet_h->setDataLocation(PacketDataLocation::CC2);

                #pragma acc parallel default(none) async(queue_h)
                {
                    #pragma acc loop gang
                    for (std::size_t n=0; n<*nTiles_d; ++n) {
                        const PacketContents*  ptrs = contents_d + n;
                        Uin_d  = ptrs->CC1_d;
                        Uout_d = ptrs->CC2_d;
                        // NOTE: On summit with PGI19.9, it appears that OpenACC is somehow determining
                        // that these should be fused into a single kernel.
                        StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                                   Uin_d, Uout_d,
                                                                                   ptrs->deltas_d);
                        StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                                  Uin_d, Uout_d,
                                                                                  ptrs->deltas_d);
                    }
                }
                break;
            case PacketDataLocation::CC2:
                packet_h->setDataLocation(PacketDataLocation::CC1);

                #pragma acc parallel default(none) async(queue_h)
                {
                    FArray4D*   Uin_d  = nullptr;
                    FArray4D*   Uout_d = nullptr;
                    #pragma acc loop gang
                    for (std::size_t n=0; n<*nTiles_d; ++n) {
                        const PacketContents*  ptrs = contents_d + n;
                        Uin_d  = ptrs->CC2_d;
                        Uout_d = ptrs->CC1_d;
                        StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                                   Uin_d, Uout_d,
                                                                                   ptrs->deltas_d);
                        StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                                  Uin_d, Uout_d,
                                                                                  ptrs->deltas_d);
                    }
                }
                break;
            default:
                throw std::logic_error("[computeLaplacianFusedKernelsWeak_packet_oacc_summit] "
                                       "Data not in CC1 or CC2");
        }
        #pragma acc wait(queue_h)
    }
}

