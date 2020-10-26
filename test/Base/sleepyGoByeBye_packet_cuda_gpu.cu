#ifndef ENABLE_CUDA_OFFLOAD
#error "This file should only be compiled if using CUDA offloading"
#endif

#include "sleepyGoByeBye.h"

#include "DataPacket.h"

#include "Flash.h"

constexpr int    NS_PER_US = 1000;
constexpr int    N_CELLS_PER_BLOCK = NXB * NYB * NZB;

__global__
void sleepyGoByeBye_cuda_gpu(const float ns) {
    __nanosleep(ns);
}

void ActionRoutines::sleepyGoByeBye_packet_cuda_gpu(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*         packet_h = dynamic_cast<DataPacket*>(dataItem_h);
    const std::size_t   nTiles_h = packet_h->nTiles();
    cudaStream_t        stream   = packet_h->stream();

    // Transfer the full amount of data for worst-case data transfer scenario
    packet_h->setVariableMask(DENS_VAR_C, ENER_VAR_C);

    constexpr float     SLEEP_NS = SLEEP_US * NS_PER_US;
    sleepyGoByeBye_cuda_gpu<<<nTiles_h,N_CELLS_PER_BLOCK,0,stream>>>(SLEEP_NS);
}

