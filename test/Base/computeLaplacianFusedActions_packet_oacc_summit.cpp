#include "computeLaplacianFused.h"

#include <Milhoja.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_DataPacket.h>

#include "Base.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit(const int tId,
                                                                     milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const PacketContents*      contents_d = packet_h->tilePointers();

    const char*  ptr_d = static_cast<char*>(packet_h->copyToGpuStart_gpu());
    const std::size_t*  nTiles_d = static_cast<std::size_t*>((void*)ptr_d);

    packet_h->setVariableMask(DENS_VAR, ENER_VAR);

    if (location != PacketDataLocation::CC1) {
        throw std::logic_error("[computeLaplacianFusedActions_packet_oacc_summit] "
                               "Input data not in CC1");
    }
    // Data will be written to Uout
    packet_h->setDataLocation(PacketDataLocation::CC2);

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        // Wait for data to arrive and then
        // launch these two independent kernels for concurrent execution
        #pragma acc wait(queue_h)
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        Uin_d  = ptrs->CC1_d;
            FArray4D*              Uout_d = ptrs->CC2_d;
            StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                       Uin_d, Uout_d,
                                                                       ptrs->deltas_d);
        }

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        Uin_d  = ptrs->CC1_d;
            FArray4D*              Uout_d = ptrs->CC2_d;
            StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                                      Uin_d, Uout_d,
                                                                      ptrs->deltas_d);
        }
    }
    #pragma acc wait(queue_h,queue2_h)

    packet_h->releaseExtraQueue(2);
}

