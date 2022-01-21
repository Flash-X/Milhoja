#include "setInitialInteriorTest.h"

#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Tile.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>

#include "Base.h"

void Simulation::setInitialInteriorTest(const int tId, milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();

    const IntVect   lo = tileDesc->lo();
    const IntVect   hi = tileDesc->hi();
    FArray4D        f  = tileDesc->data();

    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        lo, hi); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        lo, hi); 

    // We intentionally leave the GC unset.
    Real    x = 0.0;
    Real    y = 0.0;
    int     i0 = lo.I();
    int     j0 = lo.J();
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords(j);
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords(i); 

                f(i, j, k, DENS_VAR) = 1.0_wp; 
                f(i, j, k, ENER_VAR) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                       -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                       + 1.0;
            }
        }
    }
}

