#include "setInitialConditions_block.h"

#include "FArray4D.h"
#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include "Flash.h"
#include "constants.h"

void Simulation::setInitialConditions_block(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();

    // Fill in the GC data as well as we aren't doing a GC fill in any
    // of these tests
    const IntVect   loGC = tileDesc->loGC();
    const IntVect   hiGC = tileDesc->hiGC();
    FArray4D        f    = tileDesc->data();

    Real    xCoords[hiGC.I() - loGC.I() + 1];
    Real    yCoords[hiGC.J() - loGC.J() + 1];
    grid.fillCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        loGC, hiGC, xCoords); 
    grid.fillCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        loGC, hiGC, yCoords); 

    Real    x = 0.0;
    Real    y = 0.0;
    int     i0 = loGC.I();
    int     j0 = loGC.J();
    for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
        for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
            y = yCoords[j-j0];
            for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                x = xCoords[i-i0]; 

                // PROBLEM ONE
                //  Approximated exactly by second-order discretized Laplacian
                f(i, j, k, DENS_VAR_C) =   3.0*x*x*x +     x*x + x 
                                      - 2.0*y*y*y - 1.5*y*y + y
                                      + 5.0;
                // PROBLEM TWO
                //  Approximation is not exact and we know the error term exactly
                f(i, j, k, ENER_VAR_C) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                      -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                      + 1.0;
            }
        }
    }
}

