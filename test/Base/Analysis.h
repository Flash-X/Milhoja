#ifndef ANALYSIS_H__
#define ANALYSIS_H__

#include <string>

namespace Analysis {
    void   initialize(const unsigned int nBlocks);
    void   densityErrors(double* L_inf, double* meanAbsError);
    void   energyErrors(double* L_inf, double* meanAbsError);
    void   writeToFile(const std::string& filename);
    void   computeErrors_block(const int tId, void* dataItem);
    void   computeErrors_packet(const int tId, void* dataItem);
}

#endif

