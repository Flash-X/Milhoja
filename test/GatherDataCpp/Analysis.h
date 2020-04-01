#ifndef ANALYSIS_H__
#define ANALYSIS_H__

#include <string>

#include "Tile.h"

namespace Analysis {
    void   initialize(const unsigned int nBlocks);
    void   densityErrors(double* L_inf, double* meanAbsError);
    void   energyErrors(double* L_inf, double* meanAbsError);
    void   writeToFile(const std::string& filename);
    void   computeErrors(const int tId, Tile* tileDesc);
}

#endif

