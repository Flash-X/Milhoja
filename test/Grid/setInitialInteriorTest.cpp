#include "setInitialInteriorTest.h"

#include "FArray4D.h"
#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include "Flash.h"
#include "constants.h"

void Simulation::setInitialInteriorTest(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();

    // Fill in the GC data as well as we aren't doing a GC fill in any
    // of these tests
    const IntVect   lo = tileDesc->lo();
    const IntVect   hi = tileDesc->hi();
    FArray4D        f    = tileDesc->data();

    Real    xCoords[hi.I() - lo.I() + 1];
    Real    yCoords[hi.J() - lo.J() + 1];
    grid.fillCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        lo, hi, xCoords); 
    grid.fillCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        lo, hi, yCoords); 

    Real    x = 0.0;
    Real    y = 0.0;
    int     i0 = lo.I();
    int     j0 = lo.J();
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords[j-j0];
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords[i-i0]; 

                f(i, j, k, DENS_VAR_C) = 1.0_wp; 

                // PROBLEM TWO
                // Approximation is not exact and we know the error term exactly
                f(i, j, k, ENER_VAR_C) = 4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x - x
                                      -  y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                      + 1.0;
            }
        }
    }
}
