#ifndef HYDRO_ADVANCE_SOLUTION_HLL_3_PACKET_OACC_H__
#define HYDRO_ADVANCE_SOLUTION_HLL_3_PACKET_OACC_H__

#include "DataItem.h"

extern "C"   void Hydro_advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                                orchestration::DataItem* dataItem_h);

#endif

