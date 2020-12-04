#include "setInitialConditions.h"

#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

void ActionRoutines::setInitialConditions_tile_cpu(const int tId,
                                                   orchestration::DataItem* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    // Fill in the GC data as well as we aren't doing a GC fill in any
    // of these tests
    const unsigned int  level = tileDesc->level();
    const IntVect       loGC  = tileDesc->loGC();
    const IntVect       hiGC  = tileDesc->hiGC();
    FArray4D            U     = tileDesc->data();

    Grid&   grid = Grid::instance();
    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                          loGC, hiGC); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                          loGC, hiGC); 

    StaticPhysicsRoutines::setInitialConditions(loGC, hiGC, xCoords, yCoords, U);
}

