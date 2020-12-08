/**
 * \file    IO.h
 *
 * \brief 
 *
 * \todo Convert into a singleton to see if this improves the code.
 *
 * \todo How to cache block quantities by gridIndex when we have multiple
 * levels?
 *
 */

#ifndef IO_H__
#define IO_H__

#include <string>

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "DataItem.h"
#include "FArray3D.h"
#include "FArray4D.h"

#include "Flash.h"

namespace IO {
    //----- GENERAL SETUP/MANAGEMENT
    void   initialize(const std::string filename);
    void   finalize(void);

    //----- INTEGRAL QUANTITIES
    void   computeBlockIntegralQuantities(const orchestration::Real simTime,
                                          const int blockIndex,
                                          const orchestration::IntVect& lo,
                                          const orchestration::IntVect& hi,
                                          const orchestration::FArray3D& cellVolumes,
                                          const orchestration::FArray4D& solnData);
    void   computeLocalIntegralQuantities(void);
    void   writeIntegralQuantities(const orchestration::Real simTime);

    // Number of globally-summed quantities
    extern    int  nIntegralQuantities;

    extern    orchestration::Real*    localIntegralQuantities;
    extern    orchestration::Real*    globalIntegralQuantities;

    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void   computeBlockIntegralQuantities_tile_cpu(const int tId,
                                                   orchestration::DataItem* dataItem);
}

namespace io {
    //----- INTEGRAL QUANTITIES
    extern std::string            integralQuantitiesFilename;

    extern orchestration::Real*   blockIntegralQuantities_mass;
    extern orchestration::Real*   blockIntegralQuantities_xmom;
    extern orchestration::Real*   blockIntegralQuantities_ymom;
    extern orchestration::Real*   blockIntegralQuantities_zmom;
    extern orchestration::Real*   blockIntegralQuantities_ener;
    extern orchestration::Real*   blockIntegralQuantities_ke;
    extern orchestration::Real*   blockIntegralQuantities_eint;
    extern orchestration::Real*   blockIntegralQuantities_magp;
}

#endif

