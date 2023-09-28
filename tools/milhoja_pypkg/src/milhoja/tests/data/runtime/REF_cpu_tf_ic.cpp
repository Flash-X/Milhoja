#include "cpu_tf_ic.h"
#include "Tile_cpu_tf_ic.h"

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray2D.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>
#include <Milhoja_Tile.h>
#include <Milhoja_Grid.h>

#include "setInitialConditions.h"

void  cpu_tf_ic::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf_ic*  wrapper = dynamic_cast<Tile_cpu_tf_ic*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const unsigned int   MH_INTERNAL_level = tileDesc->level();
    const milhoja::IntVect   tile_lbound = tileDesc->loGC();
    const milhoja::IntVect   tile_ubound = tileDesc->hiGC();
    const milhoja::FArray1D  tile_xCenters =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::I,
            milhoja::Edge::Center,
            MH_INTERNAL_level,
            tile_lbound, tile_ubound);
    const milhoja::FArray1D  tile_yCenters =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::J,
            milhoja::Edge::Center,
            MH_INTERNAL_level,
            tile_lbound, tile_ubound);
    milhoja::FArray4D  CC_1 = tileDesc->data();

    StaticPhysicsRoutines::setInitialConditions(
                    tile_lbound,
                    tile_ubound,
                    tile_xCenters,
                    tile_yCenters,
                    CC_1);
}
