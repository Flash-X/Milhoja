#include "scaleEnergy_cpu.h"

#include "Flash.h"
#include "Grid.h"

void ThreadRoutines::scaleEnergy_cpu(const int tId,
                                     Tile* tileDesc) {
    amrex::MultiFab&    unk = Grid::instance()->unk();
    amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
    amrex::Array4<amrex::Real> const&   f = fab.array();

    amrex::Dim3 const    lo = tileDesc->lo();
    amrex::Dim3 const    hi = tileDesc->hi();
    for     (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
              f(i, j, lo.z, ENER_VAR_C) *= 3.2;
         }
    }
}

