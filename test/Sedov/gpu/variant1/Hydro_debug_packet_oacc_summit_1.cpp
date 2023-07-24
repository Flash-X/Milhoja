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
#if (MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)
    const int                  queue2_h   = packet_h->extraAsynchronousQueue(2);
#endif
#if MILHOJA_NDIM == 3
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
    packet_h->setDataLocation(PacketDataLocation::CC2);
    packet_h->setVariableMask(UNK_VARS_BEGIN, EINT_VAR);

    std::cout << "Hello from the host" << std::endl;

    #pragma acc data deviceptr(nTiles_d, contents_d, dt_d)
    {


		#pragma acc parallel loop gang default(none) async(queue_h)
        for (std::size_t n=0; n<*nTiles_d; ++n) {
            const PacketContents*  ptrs = contents_d + n;
            const FArray4D*        Uin_d    = ptrs->CC1_d;
            const FArray4D*        Uout_d    = ptrs->CC2_d;


            int     i_s = ptrs->lo_d->I();
            int     j_s = ptrs->lo_d->J();
            int     k_s = ptrs->lo_d->K();

			int     i_e = ptrs->hi_d->I();
	    	int     j_e = ptrs->hi_d->J();
	    	int     k_e = ptrs->hi_d->K();

		    #pragma acc loop vector collapse(3)
		    for         (int k=k_s; k<=k_e; ++k) {
	    	    for     (int j=j_s; j<=j_e; ++j) {
	        	    for (int i=i_s; i<=i_e; ++i) {

						
	            	    Uout_d->at(i, j, k, DENS_VAR) = Uin_d->at(i, j, k, DENS_VAR);
//						U_d->at(i, j, k, VELX_VAR) = i + j + k;
//						U_d->at(i, j, k, VELY_VAR) = i;
//						U_d->at(i, j, k, VELZ_VAR) = j;
//						U_d->at(i, j, k, ENER_VAR) = k;
//						U_d->at(i, j, k, TEMP_VAR) = 299.23;
//						U_d->at(i, j, k, EINT_VAR) = 10.8;
//						U_d->at(i, j, k, PRES_VAR) = n;

		    		}
				}
		    }
        }
    } // OpenACC data block

	packet_h->releaseExtraQueue(2);
    #pragma acc wait(queue_h)
}

