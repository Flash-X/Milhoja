#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "milhoja.h"
#include "DataPacket.h"

#include "Eos.h"
#include "Hydro.h"

#include "Sedov.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_1(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
#if NDIM >= 2
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
#endif
#if NDIM == 3
    const int                  queue3_h   = packet_h->extraAsynchronousQueue(3);
#endif
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const PacketContents*      contents_d = packet_h->tilePointers();

    const char*  ptr_d = static_cast<char*>(packet_h->copyToGpuStart_gpu());
    const std::size_t*  nTiles_d = static_cast<std::size_t*>((void*)ptr_d);
    ptr_d += sizeof(std::size_t);
    const Real*         dt_d     = static_cast<Real*>((void*)ptr_d);

    if (location != PacketDataLocation::CC1) {
        throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_1] "
                                 "Input data must be in CC1");
    }

    // This task function neither reads from nor writes to GAME.  While it does
    // read from GAMC, this variable is not written to as part of the task
    // function's work.  Therefore, GAME need not be included in the packet and
    // GAMC need not be copied back to Grid data structures as part of
    // host-side unpacking.
    //
    // Note that this optimization requires that GAMC be adjacent in memory to
    // all other variables in the packet and GAME outside of this grouping.  For
    // this test, these two variables were declared in Flash.h as the last two
    // UNK variables to accomplish this goal.
    //
    // GAMC is sent to the GPU, but does not need to be returned to the host.
    // To accomplish this, the CC1 blocks in the copy-in section are packed with
    // one more variable than the CC2 blocks packed in the copy-out section.
    // Note that the CC2 blocks are used first as "3D" scratch arrays for auxC.
    //
    // TODO: How to do the masking?  Does the setup tool/offline toolchain have
    // to determine how to assign indices to the variables so that this can
    // happen for all task actions that must filter?  Selecting the order of
    // variables in memory sounds like part of the larger optimization problem
    // as it affects all data packets.
    packet_h->setDataLocation(PacketDataLocation::CC2);
    packet_h->setVariableMask(UNK_VARS_BEGIN, EINT_VAR);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block and in place.
    #pragma acc data deviceptr(nTiles_d, contents_d, dt_d)
    {
        //----- COMPUTE FLUXES
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            FArray4D*              auxC_d = ptrs->CC2_d;

            hy::computeSoundSpeedHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                                 U_d, auxC_d);
        }

        // The X, Y, and Z fluxes each depend on the speed of sound, but can
        // be computed independently and therefore concurrently.
#if   NDIM == 1
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
        // No need for barrier since all kernels are launched on the same
        // queue for 1D case.
#elif NDIM == 2
        // Wait for data to arrive and then launch these two for concurrent
        // execution
        #pragma acc wait(queue_h)

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
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            const FArray4D*        auxC_d = ptrs->CC2_d;
            FArray4D*              flY_d  = ptrs->FCY_d;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                               ptrs->deltas_d,
                                               U_d, flY_d, auxC_d);
        }
        // BARRIER - fluxes must all be computed before updating the solution
        #pragma acc wait(queue_h,queue2_h)
        packet_h->releaseExtraQueue(2);
#elif NDIM == 3
        // Wait for data to arrive and then launch these three for concurrent
        // execution
        #pragma acc wait(queue_h)

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
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            const FArray4D*        auxC_d = ptrs->CC2_d;
            FArray4D*              flY_d  = ptrs->FCY_d;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                               ptrs->deltas_d,
                                               U_d, flY_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue3_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            const FArray4D*        auxC_d = ptrs->CC2_d;
            FArray4D*              flZ_d  = ptrs->FCZ_d;

            hy::computeFluxesHll_Z_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                               ptrs->deltas_d,
                                               U_d, flZ_d, auxC_d);
        }
        // BARRIER - fluxes must all be computed before updated the solution
        #pragma acc wait(queue_h,queue2_h,queue3_h)
        packet_h->releaseExtraQueue(2);
        packet_h->releaseExtraQueue(3);
#endif

        //----- UPDATE SOLUTIONS IN PLACE
        // U is a shared resource for all of these kernels and therefore
        // they must be launched serially.
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              Uin_d   = ptrs->CC1_d;
            FArray4D*              Uout_d  = ptrs->CC2_d;

            hy::scaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, Uin_d, Uout_d);
        }
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC2_d;
            const FArray4D*        flX_d = ptrs->FCX_d;

            hy::updateSolutionHll_FlX_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flX_d);
        }
#if NDIM >= 2
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC2_d;
            const FArray4D*        flY_d = ptrs->FCY_d;

            hy::updateSolutionHll_FlY_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flY_d);
        }
#endif
#if NDIM == 3
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC2_d;
            const FArray4D*        flZ_d = ptrs->FCZ_d;

            hy::updateSolutionHll_FlZ_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flZ_d);
        }
#endif
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC2_d;

            hy::rescaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
        }
#ifdef EINT_VAR
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC2_d;

            hy::computeEintHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
        }
#endif

        // Apply EoS on interior
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d = ptrs->CC2_d;

            Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

