#ifndef ANALYSIS_H__
#define ANALYSIS_H__

#include <string>

#include "DataItem.h"
#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"

namespace Analysis {
    void   initialize(const unsigned int nBlocks);
    void   densityErrors(double* L_inf, double* meanAbsError);
    void   energyErrors(double* L_inf, double* meanAbsError);
    void   writeToFile(const std::string& filename);
    void   computeErrors(const orchestration::IntVect& lo, const orchestration::IntVect& hi,
                         const orchestration::FArray1D& xCoords,
                         const orchestration::FArray1D& yCoords,
                         const orchestration::FArray4D& U,
                         const int idx);
}

namespace ActionRoutines {
    void   computeErrors_tile_cpu(const int tId, orchestration::DataItem* dataItem);
}

#endif

