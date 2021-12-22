#ifndef SET_INITIAL_CONDITIONS_H__
#define SET_INITIAL_CONDITIONS_H__

#include <Milhoja_IntVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_DataItem.h>

namespace StaticPhysicsRoutines {
    void setInitialConditions(const milhoja::IntVect& loGC,
                              const milhoja::IntVect& hiGC,
                              const milhoja::FArray1D& xCoords,
                              const milhoja::FArray1D& yCoords,
                              milhoja::FArray4D& U);
}

namespace ActionRoutines {
    void setInitialConditions_tile_cpu(const int tId, milhoja::DataItem* dataItem);
}

#endif

