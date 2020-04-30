#include "scaleEnergy_cpu.h"

#include "Flash.h"
#include "Grid.h"

void ThreadRoutines::scaleEnergy_cpu(const int tId,
                                     Tile* tileDesc) {
    Grid*    grid = Grid::instance();
    amrex::Geometry     geometry = grid->geometry();
    amrex::MultiFab&    unk = grid->unk();
    amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
    grid = nullptr;

    amrex::Array4<amrex::Real> const&   f = fab.array();

    amrex::Real         x = 0.0;
    amrex::Real         y = 0.0;
    const amrex::Dim3   lo = tileDesc->lo();
    const amrex::Dim3   hi = tileDesc->hi();
    for     (int j = lo.y; j <= hi.y; ++j) {
        y = geometry.CellCenter(j, 1);
        for (int i = lo.x; i <= hi.x; ++i) {
            x = geometry.CellCenter(i, 0);
            f(i, j, lo.z, ENER_VAR_C) *= 5.0 * x * y;
        }
    }
}

