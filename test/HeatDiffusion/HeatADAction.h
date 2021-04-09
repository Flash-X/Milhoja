#ifndef HEATAD_ACTION_H__
#define HEATAD_ACTION_H__

#include "DataItem.h"

namespace HeatADAction {

    void advanceSolution_tile_cpu(const int tId,
                                  orchestration::DataItem* dataItem);

};

#endif

