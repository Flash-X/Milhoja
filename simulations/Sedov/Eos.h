#ifndef EOS_H__
#define EOS_H__

#include "Grid_IntVect.h"
#include "FArray4D.h"
#include "DataItem.h"

#include "Flash.h"

namespace Eos {
    // lo/hi can be any two corners, including loGC/hiGC
    void idealGammaDensIe(const orchestration::IntVect& lo,
                          const orchestration::IntVect& hi,
                          orchestration::FArray4D& solnData);

    #pragma acc routine vector
    void idealGammaDensIe_oacc_summit(const orchestration::IntVect* lo_d,
                                      const orchestration::IntVect* hi_d,
                                      orchestration::FArray4D* solnData_d);

    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void idealGammaDensIe_packet_oacc_summit(const int tId,
                                             orchestration::DataItem* dataItem_h);
};

#endif

