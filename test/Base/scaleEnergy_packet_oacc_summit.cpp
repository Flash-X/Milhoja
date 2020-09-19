#include "scaleEnergy.h"

#include "CudaDataPacket.h"

void ActionRoutines::scaleEnergy_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;

    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       streamId_h = packet_h->stream().id;

    // TODO: This data should be included in the copyin section of the
    //       data packet.
    constexpr Real    ENERGY_SCALE_FACTOR = 5.0;
    StaticPhysicsRoutines::scaleEnergy_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                   gpuPtrs_d.xCoords, gpuPtrs_d.yCoords,
                                                   gpuPtrs_d.data, ENERGY_SCALE_FACTOR,
                                                   streamId_h);
}

