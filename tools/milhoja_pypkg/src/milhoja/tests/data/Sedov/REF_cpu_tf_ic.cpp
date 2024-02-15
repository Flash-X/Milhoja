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

#include "Eos.h"
#include "Simulation.h"

void  cpu_tf_ic::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_ic*  wrapper = dynamic_cast<Tile_cpu_tf_ic*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const unsigned int  tile_level = tileDesc->level();
    const milhoja::IntVect  tile_lbound = tileDesc->loGC();
    const milhoja::IntVect  tile_ubound = tileDesc->hiGC();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    const milhoja::FArray1D  tile_xCoords_center =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::I,
            milhoja::Edge::Center,
            tile_level,
            tile_lbound, tile_ubound);
    const milhoja::FArray1D  tile_yCoords_center =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::J,
            milhoja::Edge::Center,
            tile_level,
            tile_lbound, tile_ubound);
    const milhoja::FArray1D  tile_zCoords_center =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::K,
            milhoja::Edge::Center,
            tile_level,
            tile_lbound, tile_ubound);
    milhoja::FArray4D  CC_1 = tileDesc->data();

    sim::setInitialConditions(
                    tile_lbound,
                    tile_ubound,
                    tile_level,
                    tile_xCoords_center,
                    tile_yCoords_center,
                    tile_zCoords_center,
                    tile_deltas,
                    CC_1);
    Eos::idealGammaDensIe(
                    tile_lbound,
                    tile_ubound,
                    CC_1);
}
