#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "hy_advance_solution_hll_3_packet_oacc_tf.h"

#include "Flash.h"
#include "constants.h"
#include "DataPacket_Hydro_gpu_3.h"

//----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
extern "C" {
    void   hy_advancesolutionhll_3_packet_oacc_c2f(const int dataQ_h,
                                                   const int nTiles_h,
                                                   const int nxb_h, const int nyb_h, const int nzb_h,
                                                   const int nCcVar_h, const int nFluxVar_h,
                                                   const int* nTiles_d, const double* dt_d,
                                                   const double* deltas_start_d,
                                                   const int* lo_start_d,   const int* hi_start_d,
                                                   const int* loGC_start_d, const int* hiGC_start_d,
                                                   const double* U_start_d,
                                                   const double* auxC_start_d,
                                                   const double* faceX_start_d,
                                                   const double* faceY_start_d,
                                                   const double* faceZ_start_d);
}

//----- C TASK FUNCTION TO BE CALLED BY RUNTIME
extern "C" {
    // TODO: For the sake of C++/Fortran interoperability, this must be a
    // C-compatible interface.  Make void*?  Are const keywords part of C?
    void hy_advance_solution_hll_3_packet_oacc_tf(const int tId,
                                                  orchestration::DataItem* dataItem_h) {
        using namespace orchestration;

        DataPacket_Hydro_gpu_3*    packet_h = dynamic_cast<DataPacket_Hydro_gpu_3*>(dataItem_h);
        const PacketDataLocation   location  = packet_h->getDataLocation();
        const int                  dataQ_h   = packet_h->asynchronousQueue();
        const int                  nTiles_h  = packet_h->nTiles_host();
        int   nxb_h  = -1;
        int   nyb_h  = -1;
        int   nzb_h  = -1;
        int   nCcVar_h = -1;
        int   nFluxVar_h = NFLUXES;
        packet_h->tileSize_host(&nxb_h, &nyb_h, &nzb_h, &nCcVar_h);

        int*     nTiles_d       = packet_h->nTiles_devptr();
        // TODO: Within this layer the dt_* variables should be double since
        //       they are sent to a Fortran routine that assumes C_DOUBLE.  How to
        //       manage this?
        Real*    dt_d           = packet_h->dt_devptr();
        Real*    deltas_start_d = packet_h->deltas_devptr();
        int*     lo_start_d     = packet_h->lo_devptr();
        int*     hi_start_d     = packet_h->hi_devptr();
        int*     loGC_start_d   = packet_h->loGC_devptr();
        int*     hiGC_start_d   = packet_h->hiGC_devptr();
        Real*    U_start_d      = packet_h->U_devptr();
        Real*    auxC_start_d   = packet_h->scratchAuxC_devptr();
        Real*    faceX_start_d  = packet_h->scratchFaceX_devptr();
        Real*    faceY_start_d  = packet_h->scratchFaceY_devptr();
        Real*    faceZ_start_d  = packet_h->scratchFaceZ_devptr();

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
        // For this test, these two variables were declared in Flash.h as the
        // last two UNK variables to accomplish this goal.
        //
        // TODO: How to do the masking?  Does the setup tool/offline toolchain have
        // to determine how to assign indices to the variables so that this can
        // happen for all task functions that must filter?  Selecting the order of
        // variables in memory sounds like part of the larger optimization problem
        // as it affects all data packets.
        packet_h->setVariableMask(UNK_VARS_BEGIN_C, EINT_VAR_C);

        if (location != PacketDataLocation::CC1) {
            throw std::runtime_error("[hy_advance_solution_hll_3_packet_oacc_tf] "
                                     "Input data must be in CC1");
        }
   
        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        hy_advancesolutionhll_3_packet_oacc_c2f(dataQ_h,
                                                nTiles_h,
                                                nxb_h, nyb_h, nzb_h,
                                                nCcVar_h, nFluxVar_h,
                                                nTiles_d, dt_d,
                                                deltas_start_d,
                                                lo_start_d,   hi_start_d,
                                                loGC_start_d, hiGC_start_d,
                                                U_start_d,
                                                auxC_start_d,
                                                faceX_start_d, faceY_start_d, faceZ_start_d);
    }

}

