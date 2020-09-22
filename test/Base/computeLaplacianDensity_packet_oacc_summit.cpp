#include "computeLaplacianDensity.h"

#include "PacketDataLocation.h"
#include "CudaDataPacket.h"

#include "Flash.h"

void ActionRoutines::computeLaplacianDensity_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;
    
    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       streamId_h = packet_h->stream().id;

    // Data will be written to scratch
    FArray4D*   data_d    = nullptr;
    FArray4D*   scratch_d = nullptr;
    switch (packet_h->getDataLocation()) {
        case PacketDataLocation::CC1:
            data_d    = gpuPtrs_d.CC1;
            scratch_d = gpuPtrs_d.CC2;
            packet_h->setDataLocation(PacketDataLocation::CC2);
            break;
        case PacketDataLocation::CC2:
            data_d    = gpuPtrs_d.CC2;
            scratch_d = gpuPtrs_d.CC1;
            packet_h->setDataLocation(PacketDataLocation::CC1);
            break;
        default:
            throw std::logic_error("[computeLaplacianDensity_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }
    // If we don't limit the variable mask to DENS, then we overwrite the energy
    // data with whatever garbage was in the scratch block.
    packet_h->setVariableMask(DENS_VAR_C, DENS_VAR_C);

    StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                               data_d, scratch_d,
                                                               gpuPtrs_d.deltas,
                                                               streamId_h);
}

