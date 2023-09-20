#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_DataPacket.h>
#include "DataPacket_Hydro_gpu_1.h"

#include "Sedov.h"
#include "Eos.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void Hydro::advanceSolutionHll_packet_oacc_summit_1(const int tId,
                                                    milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket_Hydro_gpu_1* packet_h = dynamic_cast<DataPacket_Hydro_gpu_1*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);

    const int* nTiles_d = packet_h->_nTiles_d;
    const Real* dt_d = packet_h->_dt_d;
    const RealVect* deltas_d = packet_h->_tile_deltas_d;
    const IntVect* lo_d = packet_h->_tile_lo_d;
    const IntVect* hi_d = packet_h->_tile_hi_d;
    FArray4D* CC1_d = packet_h->_f4_U_d;
    FArray4D* CC2_d = packet_h->_f4_auxC_d;
    FArray4D* FCX_d = packet_h->_f4_flX_d;
    FArray4D* FCY_d = packet_h->_f4_flY_d;
    FArray4D* FCZ_d = packet_h->_f4_flZ_d;

    // This task function neither reads from nor writes to GAME.  While it does
    // read from GAMC, this variable is not written to as part of the task
    // function's work.  Therefore, GAME need not be included in the packet and
    // GAMC need not be copied back to Grid data structures as part of
    // host-side unpacking.
    //
    // Note that this optimization requires that GAMC be adjacent in memory to
    // all other variables in the packet and GAME outside of this grouping.  For
    // this test, these two variables were declared in Sedov.h as the last two
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
    
    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block and in place.
    #pragma acc data deviceptr(nTiles_d, dt_d, deltas_d, lo_d, hi_d, CC1_d, CC2_d, FCX_d, FCY_d, FCZ_d)
    {
        //----- COMPUTE FLUXES
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;

            hy::computeSoundSpeedHll_oacc_summit(lo_d + n, hi_d + n,
                                                 U_d, auxC_d);
        }

        // The X, Y, and Z fluxes each depend on the speed of sound, but can
        // be computed independently and therefore concurrently.
#if   MILHOJA_NDIM == 1
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flX_d  = FCX_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
                                               U_d, flX_d, auxC_d);
        }
        // No need for barrier since all kernels are launched on the same
        // queue for 1D case.
#elif MILHOJA_NDIM == 2
        // Wait for data to arrive and then launch these two for concurrent
        // execution
        #pragma acc wait(queue_h)

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flX_d  = FCX_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
                                               U_d, flX_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flY_d  = FCY_d + n;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
                                               U_d, flY_d, auxC_d);
        }
        // BARRIER - fluxes must all be computed before updating the solution
        #pragma acc wait(queue_h,queue2_h)
        packet_h->releaseExtraQueue(2);
#elif MILHOJA_NDIM == 3
        // Wait for data to arrive and then launch these three for concurrent
        // execution
        #pragma acc wait(queue_h)

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flX_d  = FCX_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
                                               U_d, flX_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flY_d  = FCY_d + n;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
                                               U_d, flY_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue3_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            const FArray4D*        auxC_d = CC2_d + n;
            FArray4D*              flZ_d  = FCZ_d + n;

            hy::computeFluxesHll_Z_oacc_summit(dt_d, lo_d + n, hi_d + n,
                                               deltas_d + n,
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
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              Uin_d   = CC1_d + n;
            FArray4D*              Uout_d  = CC2_d + n;

            hy::scaleSolutionHll_oacc_summit(lo_d + n, hi_d + n, Uin_d, Uout_d);
        }
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC2_d + n;
            const FArray4D*        flX_d = FCX_d + n;

            hy::updateSolutionHll_FlX_oacc_summit(lo_d + n, hi_d + n, U_d, flX_d);
        }
#if (MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC2_d + n;
            const FArray4D*        flY_d = FCY_d + n;

            hy::updateSolutionHll_FlY_oacc_summit(lo_d + n, hi_d + n, U_d, flY_d);
        }
#endif
#if MILHOJA_NDIM == 3
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC2_d + n;
            const FArray4D*        flZ_d = FCZ_d + n;

            hy::updateSolutionHll_FlZ_oacc_summit(lo_d + n, hi_d + n, U_d, flZ_d);
        }
#endif
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC2_d + n;

            hy::rescaleSolutionHll_oacc_summit(lo_d + n, hi_d + n, U_d);
        }
#ifdef EINT_VAR
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC2_d + n;

            hy::computeEintHll_oacc_summit(lo_d + n, hi_d + n, U_d);
        }
#endif

        // Apply EoS on interior
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d = CC2_d + n;

            Eos::idealGammaDensIe_oacc_summit(lo_d + n, hi_d + n, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

