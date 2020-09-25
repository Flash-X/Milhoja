#ifndef USE_OPENACC
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "computeLaplacianEnergy.h"

#include "PacketDataLocation.h"
#include "CudaDataPacket.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianEnergy_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;
    
    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       queue_h = packet_h->stream().id;

    // Data will be written to Uout
    FArray4D*   Uin_d  = nullptr;
    FArray4D*   Uout_d = nullptr;
    switch (packet_h->getDataLocation()) {
        case PacketDataLocation::CC1:
            Uin_d  = gpuPtrs_d.CC1;
            Uout_d = gpuPtrs_d.CC2;
            packet_h->setDataLocation(PacketDataLocation::CC2);
            break;
        case PacketDataLocation::CC2:
            Uin_d  = gpuPtrs_d.CC2;
            Uout_d = gpuPtrs_d.CC1;
            packet_h->setDataLocation(PacketDataLocation::CC1);
            break;
        default:
            throw std::logic_error("[computeLaplacianEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }
    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);

    StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                              Uin_d, Uout_d,
                                                              gpuPtrs_d.deltas,
                                                              queue_h);
}

