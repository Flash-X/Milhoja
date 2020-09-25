#ifndef USE_OPENACC
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "scaleEnergy.h"

#include "PacketDataLocation.h"
#include "CudaDataPacket.h"

#include "Flash.h"

void ActionRoutines::scaleEnergy_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;

    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       queue_h = packet_h->stream().id;

    // Computation done in-place 
    FArray4D*   U_d = nullptr;
    switch (packet_h->getDataLocation()) {
        case PacketDataLocation::CC1:
            U_d = gpuPtrs_d.CC1;
            break;
        case PacketDataLocation::CC2:
            U_d = gpuPtrs_d.CC2;
            break;
        default:
            throw std::logic_error("[scaleEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }
    packet_h->setVariableMask(ENER_VAR_C, ENER_VAR_C);

    // TODO: This data should be included in the copyin section of the
    //       data packet.
    constexpr Real    ENERGY_SCALE_FACTOR = 5.0;
    StaticPhysicsRoutines::scaleEnergy_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                   gpuPtrs_d.xCoords, gpuPtrs_d.yCoords,
                                                   U_d, ENERGY_SCALE_FACTOR,
                                                   queue_h);
}

