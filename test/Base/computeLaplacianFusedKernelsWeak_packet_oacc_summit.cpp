#include "computeLaplacianFused.h"

#include <Milhoja_DataItem.h>
#include <Milhoja_DataPacket.h>

#include "Base.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"

#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

void ActionRoutines::computeLaplacianFusedKernelsWeak_packet_oacc_summit(const int tId,
                                                                         milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const PacketContents*      contents_d = packet_h->tilePointers();

    const char*  ptr_d = static_cast<char*>(packet_h->copyToGpuStart_gpu());
    const std::size_t*  nTiles_d = static_cast<std::size_t*>((void*)ptr_d);

    packet_h->setVariableMask(DENS_VAR, ENER_VAR);

    if (location != PacketDataLocation::CC1) {
        throw std::logic_error("[computeLaplacianFusedKernelsWeak_packet_oacc_summit] "
                               "Imput data not in CC1");
    }
    // Data will be written to Uout
    packet_h->setDataLocation(PacketDataLocation::CC2);
 
    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        Uin_d  = ptrs->CC1_d;
            FArray4D*              Uout_d = ptrs->CC2_d;
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
    #pragma acc wait(queue_h)
}

