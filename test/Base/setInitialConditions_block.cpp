#include "setInitialConditions_block.h"

#include "Flash.h"
#include "constants.h"
#include "Grid.h"
#include "Tile.h"

using namespace orchestration;

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
    IntVect loGC = tileDesc->loGC();
    IntVect hiGC = tileDesc->hiGC();
    for     (int j = loGC[1]; j <= hiGC[1]; ++j) {
        y = geometry.CellCenter(j, 1);
        for (int i = loGC[0]; i <= hiGC[0]; ++i) {
            x = geometry.CellCenter(i, 0);
            int loGCz = 0; //TODO correct?
            // PROBLEM ONE
            //  Approximated exactly by second-order discretized Laplacian
            f(i, j, loGCz, DENS_VAR_C) =   3.0*x*x*x +     x*x + x 
                                          - 2.0*y*y*y - 1.5*y*y + y
                                          + 5.0;
            // PROBLEM TWO
            //  Approximation is not exact and we know the error term exactly
            f(i, j, loGCz, ENER_VAR_C) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                          -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                          + 1.0;
        }
    }
}

