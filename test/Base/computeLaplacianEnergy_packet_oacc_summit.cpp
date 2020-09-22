#include "computeLaplacianEnergy.h"

#include "PacketDataLocation.h"
#include "CudaDataPacket.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianEnergy_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;
    
    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       streamId_h = packet_h->stream().id;

    // This is still doing the copy back from scratch to data
    FArray4D*   data_d    = nullptr;
    FArray4D*   scratch_d = nullptr;
    switch (packet_h->getDataLocation()) {
        case PacketDataLocation::CC1:
            data_d    = gpuPtrs_d.CC1;
            scratch_d = gpuPtrs_d.CC2;
            break;
        case PacketDataLocation::CC2:
            data_d    = gpuPtrs_d.CC2;
            scratch_d = gpuPtrs_d.CC1;
            break;
        default:
            throw std::logic_error("[computeLaplacianEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }
    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    StaticPhysicsRoutines::computeLaplacianEnergy_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                              data_d, scratch_d,
                                                              gpuPtrs_d.deltas,
                                                              streamId_h);
}

