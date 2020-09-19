#include "computeLaplacianDensity.h"

#include "CudaDataPacket.h"

void ActionRoutines::computeLaplacianDensity_packet_oacc_summit(const int tId, void* dataItem_h) {
    using namespace orchestration;
    
    // TODO: This should work for any packet.
    CudaDataPacket*                 packet_h   = reinterpret_cast<CudaDataPacket*>(dataItem_h);
    const CudaDataPacket::Contents  gpuPtrs_d  = packet_h->gpuContents();
    const int                       streamId_h = packet_h->stream().id;

    StaticPhysicsRoutines::computeLaplacianDensity_oacc_summit(gpuPtrs_d.lo, gpuPtrs_d.hi,
                                                               gpuPtrs_d.data, gpuPtrs_d.scratch,
                                                               gpuPtrs_d.deltas,
                                                               streamId_h);
}

