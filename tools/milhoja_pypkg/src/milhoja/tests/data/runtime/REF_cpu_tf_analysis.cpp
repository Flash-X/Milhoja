#include "cpu_tf_analysis.h"
#include "Tile_cpu_tf_analysis.h"

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

#include "Analysis.h"

void  cpu_tf_analysis::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_analysis*  wrapper = dynamic_cast<Tile_cpu_tf_analysis*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const int  tile_gridIndex = tileDesc->gridIndex();
    const unsigned int   MH_INTERNAL_level = tileDesc->level();
    const milhoja::IntVect   tile_lo = tileDesc->lo();
    const milhoja::IntVect   tile_hi = tileDesc->hi();
    const milhoja::FArray1D  tile_xCoords_center =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::I,
            milhoja::Edge::Center,
            MH_INTERNAL_level,
            tile_lo, tile_hi);
    const milhoja::FArray1D  tile_yCoords_center =
        milhoja::Grid::instance().getCellCoords(
            milhoja::Axis::J,
            milhoja::Edge::Center,
            MH_INTERNAL_level,
            tile_lo, tile_hi);
    const milhoja::FArray4D  CC_1 = tileDesc->data();

    Analysis::computeErrors(
                    tile_lo,
                    tile_hi,
                    tile_xCoords_center,
                    tile_yCoords_center,
                    CC_1,
                    tile_gridIndex);
}
