#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "DataPacket_Hydro_gpu_3.h"
#include "OrchestrationLogger.h"

#include "Eos.h"
#include "Hydro_advanceSolutionHll_packet_oacc_summit_3.h"

#include "Flash.h"

//----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
extern "C" {
    void   hydro_advancesolutionhll_3_packet_oacc_c2f(const int dataQ_h,
                                                      const int nTiles_h,
                                                      const int* nTiles_d, const double* dt_d);
}

//----- C TASK FUNCTION TO BE CALLED BY RUNTIME
extern "C" {
    // TODO: For the sake of C++/Fortran interoperability, this must be a
    // C-compatible interface.  Make void*?  Are const keywords part of C?
    void Hydro_advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                       orchestration::DataItem* dataItem_h) {
        using namespace orchestration;
    
        DataPacket_Hydro_gpu_3*    packet_h = dynamic_cast<DataPacket_Hydro_gpu_3*>(dataItem_h);
        const int                  dataQ_h  = packet_h->asynchronousQueue();
        const int                  nTiles_h = packet_h->nTiles_host();
    
        int*                       nTiles_d = packet_h->nTiles_devptr();
        // TODO: Within this layer the dt_* variables should be double since
        //       they are sent to a Fortran routine that assumes C_DOUBLE.  How to
        //       manage this?
        Real*                      dt_d     = packet_h->dt_devptr();
    
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
   
        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        hydro_advancesolutionhll_3_packet_oacc_c2f(dataQ_h,
                                                   nTiles_h,
                                                   nTiles_d, dt_d);
    }

}

