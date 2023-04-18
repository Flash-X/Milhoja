#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_DataPacket.h>

#include "Sedov.h"
#include "Eos.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

void Hydro::debug_packet_oacc_summit_1(const int tId,
                                                    milhoja::DataItem* dataItem_h) {
    using namespace milhoja;

    DataPacket*                packet_h   = dynamic_cast<DataPacket*>(dataItem_h);

    const int                  queue_h    = packet_h->asynchronousQueue();
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
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

    packet_h->setDataLocation(PacketDataLocation::CC2);
    packet_h->setVariableMask(UNK_VARS_BEGIN, EINT_VAR);
    std::cout << "Just making sure this is the correct packet" << std::endl;
//
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
        // Wait for data to arrive and then launch these two for concurrent
        // execution
        #pragma acc wait(queue_h)

        #pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        U_d    = ptrs->CC1_d;
            const FArray4D*        auxC_d = ptrs->CC2_d;
            FArray4D*              flX_d  = ptrs->FCX_d;
            FArray4D*			   flY_d = ptrs->FCY_d;

			int     i_s = ptrs->lo_d->I();
		    int     j_s = ptrs->lo_d->J();
		    int     k_s = ptrs->lo_d->K();

		    int     i_e = ptrs->hi_d->I();
		    int     j_e = ptrs->hi_d->J();
		    int     k_e = ptrs->hi_d->K();


		    #pragma acc loop vector collapse(3)
		    for         (int k=k_s; k<=k_e;             ++k) {
		        for     (int j=j_s; j<=j_e;             ++j) {
		            for (int i=i_s; i<=i_e; 			++i) {
		                U_d->at(i, j, k, DENS_VAR) = 20.3;
		                auxC_d->at(i, j, k, DENS_VAR) = U_d->at(i, j, k, DENS_VAR);
		                flX_d->at(i, j, k, HY_DENS_FLUX) = 1;
		                flX_d->at(i, j, k, HY_XMOM_FLUX) = 2;
//        		        flX_d->at(i, j, k, HY_YMOM_FLUX) = 3;
//                		flX_d->at(i, j, k, HY_ZMOM_FLUX) = 4;
//                		flX_d->at(i, j, k, HY_ENER_FLUX) = 5;
		                flY_d->at(i, j, k, HY_DENS_FLUX) = 6; 
		            }
		        }
	    	}

			//write to auxC_d? write to FCX_d
//            hy::computeFluxesHll_X_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
//                                               ptrs->deltas_d,
//                                               U_d, flX_d, auxC_d);
        }
//        #pragma acc parallel loop gang default(none) async(queue2_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            const FArray4D*        U_d    = ptrs->CC1_d;
//            const FArray4D*        auxC_d = ptrs->CC2_d;
//            FArray4D*              flY_d  = ptrs->FCY_d;
//
//            hy::computeFluxesHll_Y_oacc_summit(dt_d, ptrs->lo_d, ptrs->hi_d,
//                                               ptrs->deltas_d,
//                                               U_d, flY_d, auxC_d);
//        }
//        // BARRIER - fluxes must all be computed before updating the solution
//        #pragma acc wait(queue_h,queue2_h)
        packet_h->releaseExtraQueue(2);
//
//        //----- UPDATE SOLUTIONS IN PLACE
//        // U is a shared resource for all of these kernels and therefore
//        // they must be launched serially.
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              Uin_d   = ptrs->CC1_d;
//            FArray4D*              Uout_d  = ptrs->CC2_d;
//
//            hy::scaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, Uin_d, Uout_d);
//        }
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              U_d   = ptrs->CC2_d;
//            const FArray4D*        flX_d = ptrs->FCX_d;
//
//            hy::updateSolutionHll_FlX_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flX_d);
//        }
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              U_d   = ptrs->CC2_d;
//            const FArray4D*        flY_d = ptrs->FCY_d;
//
//            hy::updateSolutionHll_FlY_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d, flY_d);
//        }
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              U_d   = ptrs->CC2_d;
//
//            hy::rescaleSolutionHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
//        }
//#ifdef EINT_VAR
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              U_d   = ptrs->CC2_d;
//
//            hy::computeEintHll_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
//        }
//#endif
//
//        // Apply EoS on interior
//        #pragma acc parallel loop gang default(none) async(queue_h)
//        for (std::size_t n=0; n<*nTiles_d; ++n) {
//            const PacketContents*  ptrs = contents_d + n;
//            FArray4D*              U_d = ptrs->CC2_d;
//
//            Eos::idealGammaDensIe_oacc_summit(ptrs->lo_d, ptrs->hi_d, U_d);
//        }
    } // OpenACC data block

    #pragma acc wait(queue_h)
}

