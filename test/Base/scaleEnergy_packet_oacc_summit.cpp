#include "scaleEnergy.h"

#include "PacketDataLocation.h"
#include "CudaDataPacket.h"

#include "Flash.h"

void ActionRoutines::scaleEnergy_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;

    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       streamId_h = packet_h->stream().id;

    // Computation done in-place 
    FArray4D*   data_d    = nullptr;
    switch (packet_h->getDataLocation()) {
        case PacketDataLocation::CC1:
            data_d = gpuPtrs_d.CC1;
            break;
        case PacketDataLocation::CC2:
            data_d = gpuPtrs_d.CC2;
            break;
        default:
            throw std::logic_error("[scaleEnergy_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
    }
    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    // TODO: This data should be included in the copyin section of the
    //       data packet.
    constexpr Real    ENERGY_SCALE_FACTOR = 5.0;
    StaticPhysicsRoutines::scaleEnergy_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                   gpuPtrs_d.xCoordsData, gpuPtrs_d.yCoordsData,
                                                   data_d, ENERGY_SCALE_FACTOR,
                                                   streamId_h);
}

