#include "scaleEnergy_cpu.h"

#include <cstdio>

#include <unistd.h>

#include "constants.h"

void ThreadRoutines::scaleEnergy_cpu(const int tId, Block& block) {
    unsigned int                  idx  = block.index();
    std::array<unsigned int,NDIM> lo   = block.lo();
    std::array<unsigned int,NDIM> hi   = block.hi();
    std::array<int,NDIM>          loGC = block.loGC();
    double***                     f    = block.dataPtr();

    int i0 = loGC[IAXIS];
    int j0 = loGC[JAXIS];
    for      (int i=lo[IAXIS]; i<=hi[IAXIS]; ++i) {
         for (int j=lo[JAXIS]; j<=hi[JAXIS]; ++j) {
              f[ENER_VAR][i-i0][j-j0] *= 3.2;
         }
    }
}

