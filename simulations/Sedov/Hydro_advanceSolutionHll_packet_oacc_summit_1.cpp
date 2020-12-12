#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "DataPacket.h"
#include "StreamManager.h"

#include "Eos.h"
#include "Hydro.h"

#include "Flash.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_1(const int tId,
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
            // FIXME: If we allow this request to block, the code could deadlock.  We
            // therefore, do not block in favor of aborting execution.
            // Acquire extra stream
            StreamManager& sMgr = StreamManager::instance();
            Stream         stream2 = sMgr.requestStream(false);
            const int      queue2_h = stream2.accAsyncQueue;
            if (queue2_h == NULL_ACC_ASYNC_QUEUE) {
                throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_1] "
                                         "Unable to acquire an extra asynchronous queue");
            }

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

            sMgr.releaseStream(stream2);
#elif NDIM == 3
            // Acquire extra streams
            StreamManager& sMgr = StreamManager::instance();
            Stream         stream2 = sMgr.requestStream(false);
            const int      queue2_h = stream2.accAsyncQueue;
            if (queue2_h == NULL_ACC_ASYNC_QUEUE) {
                throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_1] "
                                         "Unable to acquire second asynchronous queue");
            }
            Stream         stream3 = sMgr.requestStream(false);
            const int      queue3_h = stream3.accAsyncQueue;
            if (queue3_h == NULL_ACC_ASYNC_QUEUE) {
                throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_2] "
                                         "Unable to acquire third asynchronous queue");
            }

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

            sMgr.releaseStream(stream2);
            sMgr.releaseStream(stream3);
#endif

            //----- UPDATE SOLUTIONS IN PLACE
            // U is a shared resource for all of these kernels and therefore
            // they must be launched serially.
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;

                hy::scaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, U_d);
            }
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;
                const FArray4D*        flX_d = ptrs->FCX_d;

                hy::updateSolutionHll_FlX_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flX_d);
            }
#if NDIM >= 2
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;
                const FArray4D*        flY_d = ptrs->FCY_d;

                hy::updateSolutionHll_FlY_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flY_d);
            }
#endif
#if NDIM == 3
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;
                const FArray4D*        flZ_d = ptrs->FCZ_d;

                hy::updateSolutionHll_FlZ_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flZ_d);
            }
#endif
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;

                hy::rescaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
            }
#ifdef EINT_VAR_C
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d   = ptrs->CC1_d;

                hy::computeEintHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
            }
#endif

            // Apply EoS on interior
            #pragma acc parallel loop gang default(none) async(queue_h)
            for (std::size_t n=0; n<*nTiles_d; ++n) {
                const PacketContents*  ptrs = contents_d + n;
                FArray4D*              U_d = ptrs->CC1_d;

                Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
            }
//        } else if (location == PacketDataLocation::CC2) {
//
        } else {
            throw std::logic_error("[Hydro::advanceSolutionHll_packet_oacc_summit_1] "
                                   "Data not in CC1");
        }

    } // OpenACC data block

    #pragma acc wait(queue_h)
}

