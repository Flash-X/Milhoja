#include "setInitialConditions.h"

#include <Milhoja_IntVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>
#include <Milhoja_Grid.h>


void  sim::setInitialConditions_noRuntime(milhoja::Tile* tileDesc) {
    using namespace milhoja;

    Grid&   grid = Grid::instance();

    const unsigned int  level = tileDesc->level();
    const IntVect       lbdd = tileDesc->loGC();
    const IntVect       ubdd = tileDesc->hiGC();
    milhoja::FArray4D   U = tileDesc->data();

    const FArray1D      xCenters = grid.getCellCoords(Axis::I, Edge::Center,
                                                      level, lbdd, ubdd);
    const FArray1D      yCenters = grid.getCellCoords(Axis::J, Edge::Center,
                                                      level, lbdd, ubdd);

    StaticPhysicsRoutines::setInitialConditions(lbdd, ubdd,
                                                xCenters, yCenters, U);
}
