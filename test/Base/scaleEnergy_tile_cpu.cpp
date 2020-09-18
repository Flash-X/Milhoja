#include "scaleEnergy.h"

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Tile.h"
#include "Grid.h"

void ActionRoutines::scaleEnergy_tile_cpu(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = reinterpret_cast<Tile*>(dataItem);

    const unsigned int  level = tileDesc->level();
    const IntVect       lo    = tileDesc->lo();
    const IntVect       hi    = tileDesc->hi();
    FArray4D            f     = tileDesc->data();

    Grid&  grid = Grid::instance();
    const FArray1D   xCoords = grid.getCellCoords(Axis::I, Edge::Center,
                                                  level, lo, hi); 
    const FArray1D   yCoords = grid.getCellCoords(Axis::J, Edge::Center,
                                                  level, lo, hi); 

    // TODO: For the CPU case, should this come from the anologue of a dat
    // module?
    constexpr Real   ENERGY_SCALE_FACTOR = 5.0;
    StaticPhysicsRoutines::scaleEnergy(lo, hi, xCoords, yCoords, f, 
                                       ENERGY_SCALE_FACTOR);
}

