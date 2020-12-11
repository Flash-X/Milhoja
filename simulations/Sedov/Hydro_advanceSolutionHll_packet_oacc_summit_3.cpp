#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Hydro.h"

#include "DataPacket.h"

#include "Flash.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const std::size_t*         nTiles_d   = packet_h->nTilesGpu();
    const PacketContents*      contents_d = packet_h->tilePointers();

    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block and in place.
    #pragma acc data deviceptr(nTiles_d, contents_d)
    {
        if        (location == PacketDataLocation::CC1) {
            // Send auxC results back for storing in AMReX MFab and for
            // visualiziation.
            // FIXME: Remove this after testing auxC!
            packet_h->setDataLocation(PacketDataLocation::CC2);

            // Compute fluxes
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              auxC_d = ptrs->CC2_d;

                hy::computeSoundSpeedHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                     U_d, auxC_d);
            }
//        } else if (location == PacketDataLocation::CC2) {
//
        } else {
            throw std::logic_error("[Hydro_advanceSolutionHll_packet_oacc_summit_3] "
                                   "Data not in CC1 or CC2");
        }

    } // OpenACC data block

    #pragma acc wait(queue_h)
}

