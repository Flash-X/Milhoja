#ifndef EOS_H__
#define EOS_H__

#include <Milhoja_IntVect.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_DataItem.h>

namespace Eos {
    // lo/hi can be any two corners, including loGC/hiGC
    void idealGammaDensIe(const milhoja::IntVect& lo,
                          const milhoja::IntVect& hi,
                          milhoja::FArray4D& solnData);

    #pragma acc routine vector
    void idealGammaDensIe_oacc_summit(const milhoja::IntVect* lo_d,
                                      const milhoja::IntVect* hi_d,
                                      milhoja::FArray4D* solnData_d);

    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void idealGammaDensIe_packet_oacc_summit(const int tId,
                                             milhoja::DataItem* dataItem_h);
};

#endif

