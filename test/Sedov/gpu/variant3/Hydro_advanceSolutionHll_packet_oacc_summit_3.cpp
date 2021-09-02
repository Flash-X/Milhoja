#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include <string>

#include "DataPacket_Hydro_gpu_3.h"
#include "OrchestrationLogger.h"

#include "Eos.h"
#include "Hydro_advanceSolutionHll_packet_oacc_summit_3.h"

#include "Flash.h"

extern "C" {
    void Hydro_advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                       orchestration::DataItem* dataItem_h) {
        using namespace orchestration;
    
        DataPacket_Hydro_gpu_3*    packet_h = dynamic_cast<DataPacket_Hydro_gpu_3*>(dataItem_h);
        const int                  dataQ_h  = packet_h->asynchronousQueue();
        const std::size_t          nTiles_h = packet_h->nTiles_host();
        Real                       dt_h     = packet_h->dt_host();
    
        std::size_t*               nTiles_d = packet_h->nTiles_devptr();
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
    
        #pragma acc data deviceptr(nTiles_d, dt_d)
        {
            #pragma acc parallel async(dataQ_h)
            {
                *dt_d *= (*nTiles_d);
                *nTiles_d = 2;
            }
        }
    
        std::string   msg("");
        msg  = "[Hydro TF] nTiles = ";
        msg += std::to_string(nTiles_h);
        Logger::instance().log(msg);
    
        msg  = "[Hydro TF] dt = ";
        msg += std::to_string(dt_h);
        Logger::instance().log(msg);
    }
}

