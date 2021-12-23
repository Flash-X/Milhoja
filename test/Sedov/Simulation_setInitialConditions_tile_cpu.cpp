#include "Simulation.h"

#include <Milhoja_Grid.h>
#include <Milhoja_Tile.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>

#include "Eos.h"

void Simulation::setInitialConditions_tile_cpu(const int tId,
                                               milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    const unsigned int  level  = tileDesc->level();
    const IntVect       loGC   = tileDesc->loGC();
    const IntVect       hiGC   = tileDesc->hiGC();
    FArray4D            U      = tileDesc->data();
    const RealVect      deltas = tileDesc->deltas();

    Grid&   grid = Grid::instance();
    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                          loGC, hiGC); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                          loGC, hiGC); 
    FArray1D zCoords = grid.getCellCoords(Axis::K, Edge::Center, level,
                                          loGC, hiGC); 

    // Since we potentially have access to the analytical expression of the ICs,
    // why not use this to set GC data rather than rely on GC fill.
    sim::setInitialConditions(loGC, hiGC, level,
                              xCoords, yCoords, zCoords,
                              deltas,
                              U);
    Eos::idealGammaDensIe(loGC, hiGC, U);
}

