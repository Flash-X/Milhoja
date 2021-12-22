#ifndef ANALYSIS_H__
#define ANALYSIS_H__

#include <string>

#include <Milhoja_DataItem.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>

namespace Analysis {
    void   initialize(const unsigned int nBlocks);
    void   densityErrors(double* L_inf, double* meanAbsError);
    void   energyErrors(double* L_inf, double* meanAbsError);
    void   writeToFile(const std::string& filename);
    void   computeErrors(const milhoja::IntVect& lo, const milhoja::IntVect& hi,
                         const milhoja::FArray1D& xCoords,
                         const milhoja::FArray1D& yCoords,
                         const milhoja::FArray4D& U,
                         const int idx);
}

namespace ActionRoutines {
    void   computeErrors_tile_cpu(const int tId, milhoja::DataItem* dataItem);
}

#endif

