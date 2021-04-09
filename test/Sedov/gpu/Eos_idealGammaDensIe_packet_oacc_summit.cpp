#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Eos.h"

#include "DataPacket.h"

#include "Flash.h"

void Eos::idealGammaDensIe_packet_oacc_summit(const int tId,
                                              orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const PacketContents*      contents_d = packet_h->tilePointers();

    const char*  ptr_d = static_cast<char*>(packet_h->copyToGpuStart_gpu());
    const std::size_t*  nTiles_d = static_cast<std::size_t*>((void*)ptr_d);

    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        if        (location == PacketDataLocation::CC1) {
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d = ptrs->CC1_d;

                Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
            }
//        } else if (location == PacketDataLocation::CC2) {
//
        } else {
            throw std::logic_error("[Eos::idealGammaDensIe_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
        }

    } // OpenACC data block

    #pragma acc wait(queue_h)
}

