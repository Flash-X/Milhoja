#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Eos.h"
#include "Hydro.h"

#include "DataPacket.h"

#include "Flash.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_2(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const std::size_t*         nTiles_d   = packet_h->nTilesGpu();
    const PacketContents*      contents_d = packet_h->tilePointers();
    const Real*                dt_d       = packet_h->timeStepGpu();

    // This action routine writes the new result to CC2 without setting
    // GAME/GAMC.  Therefore, whatever data was originally in CC2 is used to
    // overwrite the true values for this variable once the packet returns to
    // the host.  Therefore, these two variables were declared in Flash.h as the
    // last two UNK variables.  By doing so, we can use the rudimentary runtime
    // variable filtering to avoid this unintended overwriting.
    // TODO: How to do the masking?  Does the setup tool/offline toolchain have
    // to determine how to assign indices to the variables so that this can
    // happen for all action routines that must filter?
    packet_h->setVariableMask(UNK_VARS_BEGIN_C, EINT_VAR_C);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already.
    // We require two CC data blocks since the velocities and energies use both
    // the old and new density.
    #pragma acc data deviceptr(nTiles_d, contents_d, dt_d)
    {
        if        (location == PacketDataLocation::CC1) {
            packet_h->setDataLocation(PacketDataLocation::CC2);

            // Compute fluxes
            // NOTE: CC2 is used solely by auxC during this stage.  It is then
            // used for computing the updated solutions in the next phase.
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

            // Update solutions using separate U data blocks so that different
            // variables can be updated simultaneously
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              Unew_d = ptrs->CC2_d;
                const FArray4D*        flX_d  = ptrs->FCX_d;
                const FArray4D*        flY_d  = ptrs->FCY_d;
                const FArray4D*        flZ_d  = ptrs->FCZ_d;
    
                hy::updateDensityHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                 U_d, Unew_d,
                                                 flX_d, flY_d, flZ_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              Unew_d = ptrs->CC2_d;
                const FArray4D*        flX_d  = ptrs->FCX_d;
                const FArray4D*        flY_d  = ptrs->FCY_d;
                const FArray4D*        flZ_d  = ptrs->FCZ_d;

                hy::updateVelxHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                              U_d, Unew_d,
                                              flX_d, flY_d, flZ_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              Unew_d = ptrs->CC2_d;
                const FArray4D*        flX_d  = ptrs->FCX_d;
                const FArray4D*        flY_d  = ptrs->FCY_d;
                const FArray4D*        flZ_d  = ptrs->FCZ_d;

                hy::updateVelyHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                              U_d, Unew_d,
                                              flX_d, flY_d, flZ_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              Unew_d = ptrs->CC2_d;
                const FArray4D*        flX_d  = ptrs->FCX_d;
                const FArray4D*        flY_d  = ptrs->FCY_d;
                const FArray4D*        flZ_d  = ptrs->FCZ_d;

                hy::updateVelzHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                              U_d, Unew_d,
                                              flX_d, flY_d, flZ_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                const FArray4D*        U_d    = ptrs->CC1_d;
                FArray4D*              Unew_d = ptrs->CC2_d;
                const FArray4D*        flX_d  = ptrs->FCX_d;
                const FArray4D*        flY_d  = ptrs->FCY_d;
                const FArray4D*        flZ_d  = ptrs->FCZ_d;

                hy::updateEnergyHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                U_d, Unew_d,
                                                flX_d, flY_d, flZ_d);
            }
#ifdef EINT_VAR_C
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs   = contents_d + n;
                FArray4D*              Unew_d = ptrs->CC2_d;

                hy::computeEintHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, Unew_d);
            }
#endif

            // Apply EoS on interior
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              Unew_d = ptrs->CC2_d;

                Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, Unew_d);
            }
//        } else if (location == PacketDataLocation::CC2) {
//
        } else {
            throw std::logic_error("[Hydro::advanceSolutionHll_packet_oacc_summit_2] "
                                   "Data not in CC1");
        }

    } // OpenACC data block

    #pragma acc wait(queue_h)
}

