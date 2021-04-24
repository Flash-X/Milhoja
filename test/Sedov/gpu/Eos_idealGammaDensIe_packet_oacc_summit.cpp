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

    // This task function neither reads from nor writes to GAME or GAMC.
    // Therefore, these two varaibles need not be included in the packet and
    // certainly need not be copied back to Grid data structures as part of
    // host-side unpacking.
    //
    // Note that this optimization requires that GAMC and GAME not be intermixed
    // with any of the variables that must be included in the data packet.  For
    // this test, these two variables were declared in Flash.h as the last two
    // UNK variables to accomplish this goal.
    //
    // TODO: This task function's packet could be even smaller.
    //
    // TODO: How to do the masking?  Does the setup tool/offline toolchain have
    // to determine how to assign indices to the variables so that this can
    // happen for all task actions that must filter?  Selecting the order of
    // variables in memory sounds like part of the larger optimization problem
    // as it affects all data packets.
    packet_h->setVariableMask(UNK_VARS_BEGIN_C, EINT_VAR_C);

    if (location != PacketDataLocation::CC1) {
        throw std::runtime_error("[Eos::idealGammaDensIe_packet_oacc_summit] "
                                 "Input data must be in CC1");
    }

    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d = ptrs->CC1_d;

            Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

