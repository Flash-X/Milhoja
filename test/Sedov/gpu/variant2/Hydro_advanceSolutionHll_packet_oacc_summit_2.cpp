#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_DataPacket.h>
#include "DataPacket_Hydro_gpu_2.h"

#include "Sedov.h"
#include "Eos.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void Hydro::advanceSolutionHll_packet_oacc_summit_2(const int tId,
                                                    milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket_Hydro_gpu_2*    packet_h   = dynamic_cast<DataPacket_Hydro_gpu_2*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
    const int                  queue3_h   = packet_h->extraAsynchronousQueue(3);
    const int                  queue4_h   = packet_h->extraAsynchronousQueue(4);

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

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already.
    // We require two CC data blocks since the velocities and energies use both
    // the old and new density.
    #pragma acc data deviceptr(nTiles_d, dt_d, CC1_d, CC2_d, lo_d, hi_d, deltas_d, FCX_d, FCY_d, FCZ_d)
    {
        //----- COMPUTE FLUXES
        // NOTE: CC2 is used solely by auxC during this stage.  It is then
        // used for computing the updated solutions in the next phase.
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::computeSoundSpeedHll_oacc_summit(lo, hi,
                                                 U_d, auxC_d);
        }

       // The X, Y, and Z fluxes each depend on the speed of sound, but can
        // be computed independently and therefore concurrently.
#if   MILHOJA_NDIM == 1
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flX_d = FCX_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo, hi,
                                               deltas,
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
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flX_d = FCX_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flX_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d    = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flY_d  = FCY_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flY_d, auxC_d);
        }
        // BARRIER - fluxes must all be computed before updating the solution
        #pragma acc wait(queue_h,queue2_h)
#elif MILHOJA_NDIM == 3
        // Wait for data to arrive and then launch these three for concurrent
        // execution
        #pragma acc wait(queue_h)

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d    = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flX_d  = FCX_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_X_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flX_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flY_d = FCY_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flY_d, auxC_d);
        }
        #pragma acc parallel loop gang default(none) async(queue3_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flZ_d = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            const RealVect* deltas = deltas_d + n;

            hy::computeFluxesHll_Z_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flZ_d, auxC_d);
        }
        // BARRIER - fluxes must all be computed before updated the solution
        #pragma acc wait(queue_h,queue2_h,queue3_h)
#endif

        //----- UPDATE SOLUTIONS
        // Update solutions using separate U data blocks so that different
        // variables can be updated simultaneously
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              Unew_d = CC2_d + n;
            const FArray4D*        flX_d  = FCX_d + n;
            const FArray4D*        flY_d  = FCY_d + n;
            const FArray4D*        flZ_d  = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
    
            hy::updateDensityHll_oacc_summit(lo, hi,
                                             U_d, Unew_d,
                                             flX_d, flY_d, flZ_d);
        }

        // The velocities and energy depend on density, but can be updated
        // independently and therefore concurrently.
        #pragma acc wait(queue_h)

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              Unew_d = CC2_d + n;
            const FArray4D*        flX_d  = FCX_d + n;
            const FArray4D*        flY_d  = FCY_d + n;
            const FArray4D*        flZ_d  = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::updateVelxHll_oacc_summit(lo, hi,
                                          U_d, Unew_d,
                                          flX_d, flY_d, flZ_d);
        }
        #pragma acc parallel loop gang default(none) async(queue2_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              Unew_d = CC2_d + n;
            const FArray4D*        flX_d  = FCX_d + n;
            const FArray4D*        flY_d  = FCY_d + n;
            const FArray4D*        flZ_d  = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::updateVelyHll_oacc_summit(lo, hi,
                                          U_d, Unew_d,
                                          flX_d, flY_d, flZ_d);
        }
        #pragma acc parallel loop gang default(none) async(queue3_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              Unew_d = CC2_d + n;
            const FArray4D*        flX_d  = FCX_d + n;
            const FArray4D*        flY_d  = FCY_d + n;
            const FArray4D*        flZ_d  = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::updateVelzHll_oacc_summit(lo, hi,
                                          U_d, Unew_d,
                                          flX_d, flY_d, flZ_d);
        }
        #pragma acc parallel loop gang default(none) async(queue4_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D*        U_d    = CC1_d + n;
            FArray4D*              Unew_d = CC2_d + n;
            const FArray4D*        flX_d  = FCX_d + n;
            const FArray4D*        flY_d  = FCY_d + n;
            const FArray4D*        flZ_d  = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::updateEnergyHll_oacc_summit(lo, hi,
                                            U_d, Unew_d,
                                            flX_d, flY_d, flZ_d);
        }

        // BARRIER - Remaining computations depend on updated solutions
        #pragma acc wait(queue_h,queue2_h,queue3_h,queue4_h)

        // Release streams as early as possible
        packet_h->releaseExtraQueue(2);
        packet_h->releaseExtraQueue(3);
        packet_h->releaseExtraQueue(4);

#ifdef EINT_VAR
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D* Unew_d = CC2_d + n;
            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            hy::computeEintHll_oacc_summit(lo, hi, Unew_d);
        }
#endif

        // Apply EoS on interior
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              Unew_d = CC2_d + n;
            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;
            Eos::idealGammaDensIe_oacc_summit(lo, hi, Unew_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

