#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Eos.h"
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
    const Real*                dt_d       = packet_h->timeStepGpu();

    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block and in place.
    #pragma acc data deviceptr(nTiles_d, contents_d, dt_d)
    {
        if        (location == PacketDataLocation::CC1) {
            // Compute fluxes
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              auxC_d = ptrs->CC2_d;

                hy::computeSoundSpeedHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                     U_d, auxC_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                const FArray4D*        auxC_d = ptrs->CC2_d;
                FArray4D*              flX_d  = ptrs->FCX_d;

                hy::computeFluxesHll_X_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                                   ptrs->deltas_d,
                                                   U_d, flX_d, auxC_d);
            }
#if NDIM >= 2
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                const FArray4D*        auxC_d = ptrs->CC2_d;
                FArray4D*              flY_d  = ptrs->FCY_d;

                hy::computeFluxesHll_Y_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                                   ptrs->deltas_d,
                                                   U_d, flY_d, auxC_d);
            }
#endif
#if NDIM == 3
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                const FArray4D*        auxC_d = ptrs->CC2_d;
                FArray4D*              flZ_d  = ptrs->FCZ_d;

                hy::computeFluxesHll_Z_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                                   ptrs->deltas_d,
                                                   U_d, flZ_d, auxC_d);
            }
#endif

            // Update solutions
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;
                const FArray4D*        flX_d = ptrs->FCX_d;
                const FArray4D*        flY_d = ptrs->FCY_d;
                const FArray4D*        flZ_d = ptrs->FCZ_d;

                hy::updateSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                  U_d, flX_d, flY_d, flZ_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d = ptrs->CC1_d;

                Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
            }
//        } else if (location == PacketDataLocation::CC2) {
//
        } else {
            throw std::logic_error("[computeLaplacianDensity_packet_oacc_summit] "
                                   "Data not in CC1 or CC2");
        }

    } // OpenACC data block

    #pragma acc wait(queue_h)
}

