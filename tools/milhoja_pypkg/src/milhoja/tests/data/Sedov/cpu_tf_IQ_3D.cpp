#include "cpu_tf_IQ.h"
#include "Tile_cpu_tf_IQ.h"

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

#include "Io.h"

void  cpu_tf_IQ::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf_IQ*  wrapper = dynamic_cast<Tile_cpu_tf_IQ*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    milhoja::Real* ptr_mh_internal_volumes =
        static_cast<milhoja::Real*>(Tile_cpu_tf_IQ::_mh_internal_volumes_)
        + Tile_cpu_tf_IQ::_MH_INTERNAL_VOLUMES_SIZE_ * threadId;
    Grid::instance().fillCellVolumes(
        tileDesc->level(),
        tileDesc->lo(),
        tileDesc->hi(),
        ptr_mh_internal_volumes);
    milhoja::FArray3D  tile_cellVolumes{ptr_mh_internal_volumes, tileDesc->lo(), tileDesc->hi()};
    milhoja::FArray4D  CC_1 = tileDesc->data();

    Io::instance().computeIntegralQuantitiesByBlock(
                    threadId,
                    tile_lo,
                    tile_hi,
                    tile_cellVolumes,
                    CC_1);
}
