#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_DataPacket.h>

#include "Sedov.h"
#include "Eos.h"

#include "DataPacket_Hydro_gpu_3.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void Hydro::advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                    milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);
    DataPacket_Hydro_gpu_3*    hydro_pkt = dynamic_cast<DataPacket_Hydro_gpu_3*>(packet_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
#if MILHOJA_NDIM == 3
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
    const int                  queue3_h   = packet_h->extraAsynchronousQueue(3);
#endif
    // const PacketContents*      contents_d = packet_h->tilePointers();

	const int* nTiles_d = hydro_pkt->_nTiles_d;
	const Real* dt_d = hydro_pkt->_dt_d;
    RealVect* deltas_d = hydro_pkt->_deltas_d;
    IntVect* lo_d = hydro_pkt->_lo_d;
    IntVect* hi_d  = hydro_pkt->hi_d;
    FArray4D* CC1_d = hydro_pkt->_f4_U_d;
    FArray4D* CC2_d = hydro_pkt->_f4_auxC_d;
    FArray4D* FCX_d = hydro_pkt->_f4_flX_d;
    FArray4D* FCY_d = hydro_pkt->_f4_flY_d;
    FArray4D* FCZ_d = hydro_pkt->_f4_flZ_d;
    

    // This task function neither reads from nor writes to GAME.  While it does
    // read from GAMC, this variable is not written to as part of the task
    // function's work.  Therefore, GAME need not be included in the packet and
    // GAMC need not be copied back to Grid data structures as part of
    // host-side unpacking.
    // 
    // For this task function, the following masking of variables is not an
    // optimization.  Without this masking, whatever data was originally in CC2
    // would be used to overwrite true values for these two variables during
    // host-side unpacking.  
    //
    // Note that to avoid such overwriting, GAMC must be adjacent in memory
    // to all other variables in the packet and GAME outside of this grouping.
    // For this test, these two variables were declared in Sedov.h as the
    // last two UNK variables to accomplish this goal.
    //
    // TODO: How to do the masking?  Does the setup tool/offline toolchain have
    // to determine how to assign indices to the variables so that this can
    // happen for all task functions that must filter?  Selecting the order of
    // variables in memory sounds like part of the larger optimization problem
    // as it affects all data packets.

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block and in place.
    #pragma acc data deviceptr(nTiles_d, deltas_d, lo_d, hi_d, CC1_d, CC2_d, FCX_d, FCY_d, FCZ_d, dt_d)
    {
        //----- COMPUTE FLUXES
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              auxC_d = CC2_d + n;

            hy::computeSoundSpeedHll_oacc_summit(lo_d, hi_d,
                                                 U_d, auxC_d);
        }

        // The X, Y, and Z fluxes each depend on the speed of sound, but can
        // be computed independently and therefore concurrently.
#if   MILHOJA_NDIM == 1
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
#elif MILHOJA_NDIM == 2
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flX_d  = FCX_d + n;
            FArray4D*              flY_d  = FCY_d + n;

            // It seems like for small 2D blocks, fusing kernels is more
            // efficient than fusing actions (i.e. running the two kernels
            // concurrently).  Too much work for the GPU?  Too much overhead
            // from the stream sync (i.e. OpenACC wait)?
            hy::computeFluxesHll_X_oacc_summit(dt_d, lo_d, hi_d,
                                               deltas_d,
                                               U_d, flX_d, auxC_d);
            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo_d, hi_d,
                                               deltas_d,
                                               U_d, flY_d, auxC_d);
        }
#elif MILHOJA_NDIM == 3
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
            FArray4D*              U_d   = CC1_d + n;
            const FArray4D*        flX_d = FCX_d + n;
            const FArray4D*        flY_d = FCY_d + n;
            const FArray4D*        flZ_d = FCZ_d + n;

            // NOTE: If MILHOJA_NDIM < 3, then some of the FC[YZ]_d will be garbage.
            //       We therefore assume that this routine will not use
            //       those fluxes associated with axes "above" MILHOJA_NDIM.
            hy::updateSolutionHll_oacc_summit(lo_d, hi_d,
                                              U_d, flX_d, flY_d, flZ_d);
        }
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d = CC1_d + n;

            Eos::idealGammaDensIe_oacc_summit(lo_d, hi_d, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

