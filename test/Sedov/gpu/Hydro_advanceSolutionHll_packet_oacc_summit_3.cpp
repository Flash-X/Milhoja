#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "DataPacket.h"
#include "Backend.h"

#include "Eos.h"
#include "Hydro.h"

#include "Flash.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                    orchestration::DataItem* dataItem_h) {
    using namespace orchestration;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
#if NDIM == 3
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
    const int                  queue3_h   = packet_h->extraAsynchronousQueue(3);
#endif
    const PacketDataLocation   location   = packet_h->getDataLocation();
    const PacketContents*      contents_d = packet_h->tilePointers();

    const char*  ptr_d = static_cast<char*>(packet_h->copyToGpuStart_gpu());
    const std::size_t*  nTiles_d = static_cast<std::size_t*>((void*)ptr_d);
    ptr_d += sizeof(std::size_t);
    const Real*         dt_d     = static_cast<Real*>((void*)ptr_d);

    packet_h->setVariableMask(UNK_VARS_BEGIN_C, UNK_VARS_END_C);

    if (location != PacketDataLocation::CC1) {
        throw std::runtime_error("[Hydro::advanceSolutionHll_packet_oacc_summit_3] "
                                 "Input data must be in CC1");
    }

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
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            const FArray4D*        auxC_d = ptrs->CC2_d;
            FArray4D*              flX_d  = ptrs->FCX_d;
            FArray4D*              flY_d  = ptrs->FCY_d;

            // It seems like for small 2D blocks, fusing kernels is more
            // efficient than fusing actions (i.e. running the two kernels
            // concurrently).  Too much work for the GPU?  Too much overhead
            // from the stream sync (i.e. OpenACC wait)?
            hy::computeFluxesHll_X_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                               ptrs->deltas_d,
                                               U_d, flX_d, auxC_d);
            hy::computeFluxesHll_Y_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
                                               ptrs->deltas_d,
                                               U_d, flY_d, auxC_d);
        }
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
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d   = ptrs->CC1_d;
            const FArray4D*        flX_d = ptrs->FCX_d;
            const FArray4D*        flY_d = ptrs->FCY_d;
            const FArray4D*        flZ_d = ptrs->FCZ_d;

            // NOTE: If NDIM < 3, then some of the FC[YZ]_d will be garbage.
            //       We therefore assume that this routine will not use
            //       those fluxes associated with axes "above" NDIM.
            hy::updateSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d,
                                              U_d, flX_d, flY_d, flZ_d);
        }
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            FArray4D*              U_d = ptrs->CC1_d;

            Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

