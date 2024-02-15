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

void  cpu_tf_IQ::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_IQ*  wrapper = dynamic_cast<Tile_cpu_tf_IQ*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const unsigned int   MH_INTERNAL_level = tileDesc->level();
    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    milhoja::Real*   MH_INTERNAL_cellVolumes_ptr =
        static_cast<milhoja::Real*>(Tile_cpu_tf_IQ::MH_INTERNAL_cellVolumes_)
        + Tile_cpu_tf_IQ::MH_INTERNAL_CELLVOLUMES_SIZE_ * threadIndex;
    milhoja::Grid::instance().fillCellVolumes(
        MH_INTERNAL_level,
        tile_lo,
        tile_hi,
        MH_INTERNAL_cellVolumes_ptr);
    const milhoja::FArray3D  tile_cellVolumes{
            MH_INTERNAL_cellVolumes_ptr,
            tile_lo,
            tile_hi};
    const milhoja::FArray4D  CC_1 = tileDesc->data();

    Io::instance().computeIntegralQuantitiesByBlock(
                    threadIndex,
                    tile_lo,
                    tile_hi,
                    tile_cellVolumes,
                    CC_1);
}
