#include "scaleEnergy_cpu.h"

#include "constants.h"

void ThreadRoutines::scaleEnergy_cpu(const int tId,
                                     Tile& tileDesc) {
    amrex::Array4<amrex::Real> const&   f = tileDesc.data();

    amrex::Dim3 const    lo = tileDesc.lo();
    amrex::Dim3 const    hi = tileDesc.hi();
    for     (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
              f(i, j, lo.z, ENER_VAR) *= 3.2;
         }
    }
}

