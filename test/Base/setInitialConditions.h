#ifndef SET_INITIAL_CONDITIONS_H__
#define SET_INITIAL_CONDITIONS_H__

#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace StaticPhysicsRoutines {
    void setInitialConditions(const orchestration::IntVect& loGC,
                              const orchestration::IntVect& hiGC,
                              const orchestration::FArray1D& xCoords,
                              const orchestration::FArray1D& yCoords,
                              orchestration::FArray4D& U);
}

namespace ActionRoutines {
    void setInitialConditions_tile_cpu(const int tId, orchestration::DataItem* dataItem);
}

#endif

