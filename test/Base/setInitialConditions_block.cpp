#include "setInitialConditions_block.h"

#include "Flash.h"
#include "constants.h"
#include "Grid.h"
#include "Tile.h"

void Simulation::setInitialConditions_block(const int tId, void* dataItem) {
    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&    grid = Grid::instance();
    amrex::Geometry     geometry = grid.geometry();
    amrex::MultiFab&    unk = grid.unk();
    amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];

    // TODO: Make getting data ptr in C++ a method in Tile?
    amrex::Array4<amrex::Real> const&   f = fab.array();

    // Fill in the GC data as well as we aren't doing a GC fill in any
    // of these tests
    amrex::Real   x = 0.0;
    amrex::Real   y = 0.0;
    const amrex::Dim3 loGC = tileDesc->loGC();
    const amrex::Dim3 hiGC = tileDesc->hiGC();
    for     (int j = loGC.y; j <= hiGC.y; ++j) {
        y = geometry.CellCenter(j, 1);
        for (int i = loGC.x; i <= hiGC.x; ++i) {
            x = geometry.CellCenter(i, 0);
            // PROBLEM ONE
            //  Approximated exactly by second-order discretized Laplacian
            f(i, j, loGC.z, DENS_VAR_C) =   3.0*x*x*x +     x*x + x 
                                          - 2.0*y*y*y - 1.5*y*y + y
                                          + 5.0;
            // PROBLEM TWO
            //  Approximation is not exact and we know the error term exactly
            f(i, j, loGC.z, ENER_VAR_C) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                          -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                          + 1.0;
        }
    }
}

