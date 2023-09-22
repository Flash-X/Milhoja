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

void  cpu_tf_analysis::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf_analysis*  wrapper = dynamic_cast<Tile_cpu_tf_analysis*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const int  tile_gridIndex = tileDesc->gridIndex();
    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::FArray1D  tile_xCenters = milhoja::Grid::instance().getCellCoords(
		milhoja::Axis::I,
		milhoja::Edge::Center,
		tileDesc->level(), 
		tileDesc->loGC(), 
		tileDesc->hiGC()
	);
    const milhoja::FArray1D  tile_yCenters = milhoja::Grid::instance().getCellCoords(
		milhoja::Axis::J,
		milhoja::Edge::Center,
		tileDesc->level(), 
		tileDesc->loGC(), 
		tileDesc->hiGC()
	);
    milhoja::FArray4D  CC_1 = tileDesc->data();

    Analysis::computeErrors(
                    tile_lo,
                    tile_hi,
                    tile_xCenters,
                    tile_yCenters,
                    CC_1,
                    tile_gridIndex);
}