#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_DataPacket.h>
#include "DataPacket_Hydro_gpu_3.h"

#include "Sedov.h"
#include "Eos.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void Hydro::advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                    milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket_Hydro_gpu_3*    packet_h   = dynamic_cast<DataPacket_Hydro_gpu_3*>(dataItem_h);
    const int                  queue_h    = packet_h->asynchronousQueue();

	const int* nTiles_d = packet_h->_nTiles_d;
	const Real* dt_d = packet_h->_dt_d;
    const RealVect* deltas_d = packet_h->_deltas_d;
    const IntVect* lo_d = packet_h->_lo_d;
    const IntVect* hi_d = packet_h->_hi_d;
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
    #pragma acc data deviceptr(nTiles_d, dt_d, deltas_d, lo_d, hi_d, CC1_d, CC2_d, FCX_d, FCY_d, FCZ_d)
    {
        //----- COMPUTE FLUXES
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            hy::computeSoundSpeedHll_oacc_summit(lo, hi,
                                                 U_d, auxC_d);
        }

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            const FArray4D* U_d = CC1_d + n;
            FArray4D* auxC_d = CC2_d + n;
            FArray4D* flX_d = FCX_d + n;
            FArray4D* flY_d = FCY_d + n;

            const RealVect* deltas = deltas_d + n;
            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

//            // It seems like for small 2D blocks, fusing kernels is more
//            // efficient than fusing actions (i.e. running the two kernels
//            // concurrently).  Too much work for the GPU?  Too much overhead
//            // from the stream sync (i.e. OpenACC wait)?
            hy::computeFluxesHll_X_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flX_d, auxC_d);
            hy::computeFluxesHll_Y_oacc_summit(dt_d, lo, hi,
                                               deltas,
                                               U_d, flY_d, auxC_d);
        }

        //----- UPDATE SOLUTIONS IN PLACE
        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d   = CC1_d + n;
            const FArray4D*        flX_d = FCX_d + n;
            const FArray4D*        flY_d = FCY_d + n;
            const FArray4D*        flZ_d = FCZ_d + n;

            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            // NOTE: If MILHOJA_NDIM < 3, then some of the FC[YZ]_d will be garbage.
            //       We therefore assume that this routine will not use
            //       those fluxes associated with axes "above" MILHOJA_NDIM.
            hy::updateSolutionHll_oacc_summit(lo, hi,
                                              U_d, flX_d, flY_d, flZ_d);
        }

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (int n=0; n<*nTiles_d; ++n) {
            FArray4D*              U_d = CC1_d + n;
            const IntVect* lo = lo_d + n;
            const IntVect* hi = hi_d + n;

            Eos::idealGammaDensIe_oacc_summit(lo, hi, U_d);
        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}
