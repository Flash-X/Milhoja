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

    Real    xCoords[hi[Axis::I] - lo[Axis::I] + 1];
    Real    yCoords[hi[Axis::J] - lo[Axis::J] + 1];
    grid.fillCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        lo, hi, xCoords); 
    grid.fillCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        lo, hi, yCoords); 

    Real    x = 0.0;
    Real    y = 0.0;
    int     i0 = lo[Axis::I];
    int     j0 = lo[Axis::J];
//    for         (int k = lo[Axis::K]; k <= hi[Axis::K]; ++k) {
        for     (int j = lo[Axis::J]; j <= hi[Axis::J]; ++j) {
            y = yCoords[j-j0];
            for (int i = lo[Axis::I]; i <= hi[Axis::I]; ++i) {
                x = xCoords[i-i0];
                f(i, j, ENER_VAR_C) *= 5.0 * x * y;
            }
        }
//    }
}

