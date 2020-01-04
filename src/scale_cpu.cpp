#include "scale_cpu.h"

#include <cstdio>

#include <unistd.h>

#include "constants.h"

void ThreadRoutines::scale_cpu(const unsigned int tId,
                               const std::string& name,
                               Block& block) {
    static const unsigned int DENS_VAR = 0;
    static const unsigned int ENER_VAR = 1;

    unsigned int                  idx  = block.index();
    std::array<unsigned int,NDIM> lo   = block.lo();
    std::array<unsigned int,NDIM> hi   = block.hi();
    std::array<int,NDIM>          loGC = block.loGC();
    double***                     f    = block.dataPtr();

    int i0 = loGC[IAXIS];
    int j0 = loGC[JAXIS];
    for      (int i=lo[IAXIS]; i<=hi[IAXIS]; ++i) {
         for (int j=lo[JAXIS]; j<=hi[JAXIS]; ++j) {
              f[DENS_VAR][i-i0][j-j0] *= 2.1;
              f[ENER_VAR][i-i0][j-j0] *= 3.2;
         }
    }

#ifdef VERBOSE
    printf("[%s / Thread %d] Scaled block %d\n", name.c_str(), tId, idx);
#endif
}

