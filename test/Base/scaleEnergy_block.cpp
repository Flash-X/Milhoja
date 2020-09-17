#include "scaleEnergy_block.h"

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "FArray4D.h"
#include "Grid.h"
#include "Tile.h"

#include "Flash.h"

void ThreadRoutines::scaleEnergy_block(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile* tileDesc = static_cast<Tile*>(dataItem);

    Grid&  grid = Grid::instance();

    const IntVect   lo = tileDesc->lo();
    const IntVect   hi = tileDesc->hi();
    FArray4D        f  = tileDesc->data();

    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        lo, hi); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        lo, hi); 

    Real    x = 0.0;
    Real    y = 0.0;
    int     i0 = lo.I();
    int     j0 = lo.J();
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords(j);
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords(i);
                f(i, j, k, ENER_VAR_C) *= 5.0 * x * y;
            }
        }
    }
}

