#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "DataPacket.h"
#include "StreamManager.h"

#include "Eos.h"
#include "Hydro.h"

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

    //----- ACQUIRE EXTRA STREAMS
    // FIXME: If we allow this request to block, the code could deadlock.  We
    // therefore, do not block in favor of aborting execution.  Should the data
    // packets take a parameter at instantiation that specifies how many streams
    // it should acquire eagerly?
    // Acquire extra stream
    StreamManager& sMgr = StreamManager::instance();

    Stream         stream2 = sMgr.requestStream(false);
    const int      queue2_h = stream2.accAsyncQueue;
    if (queue2_h == NULL_ACC_ASYNC_QUEUE) {
        throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_2] "
                                 "Unable to acquire second asynchronous queue");
    }

    Stream         stream3 = sMgr.requestStream(false);
    const int      queue3_h = stream3.accAsyncQueue;
    if (queue3_h == NULL_ACC_ASYNC_QUEUE) {
        throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_2] "
                                 "Unable to acquire third asynchronous queue");
    }

    Stream         stream4 = sMgr.requestStream(false);
    const int      queue4_h = stream4.accAsyncQueue;
    if (queue4_h == NULL_ACC_ASYNC_QUEUE) {
        throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_2] "
                                 "Unable to acquire fourth asynchronous queue");
    }

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already.
    // We require two CC data blocks since the velocities and energies use both
    // the old and new density.
    #pragma acc data deviceptr(nTiles_d, contents_d, dt_d)
    {
        if        (location == PacketDataLocation::CC1) {
            packet_h->setDataLocation(PacketDataLocation::CC2);

            //----- COMPUTE FLUXES
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
#endif

            //----- UPDATE SOLUTIONS
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

            // The velocities and energy depend on density, but can be updated
            // independently and therefore concurrently.
            #pragma acc wait(queue_h)

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
            #pragma acc parallel loop gang default(none) async(queue2_h)
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
            #pragma acc parallel loop gang default(none) async(queue3_h)
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
            #pragma acc parallel loop gang default(none) async(queue4_h)
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

            // BARRIER - Remaining computations depend on updated solutions
            #pragma acc wait(queue_h,queue2_h,queue3_h,queue4_h)

            // Release streams as early as possible
            sMgr.releaseStream(stream2);
            sMgr.releaseStream(stream3);
            sMgr.releaseStream(stream4);

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

