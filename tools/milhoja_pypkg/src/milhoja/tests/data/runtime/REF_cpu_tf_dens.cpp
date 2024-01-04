#include "cpu_tf_dens.h"
#include "Tile_cpu_tf_dens.h"

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

#include "computeLaplacianDensity.h"

void  cpu_tf_dens::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_dens*  wrapper = dynamic_cast<Tile_cpu_tf_dens*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    milhoja::FArray4D  CC_1 = tileDesc->data();
    const milhoja::IntVect    lo_scratch_base_op1_scratch3D = milhoja::IntVect{LIST_NDIM(tile_lo.I(),
                                       tile_lo.J(),
                                       tile_lo.K())};
    const milhoja::IntVect    hi_scratch_base_op1_scratch3D = milhoja::IntVect{LIST_NDIM(tile_hi.I(),
                                       tile_hi.J(),
                                       tile_hi.K())};
    milhoja::Real* ptr_scratch_base_op1_scratch3D = 
             static_cast<milhoja::Real*>(Tile_cpu_tf_dens::scratch_base_op1_scratch3D_)
            + Tile_cpu_tf_dens::SCRATCH_BASE_OP1_SCRATCH3D_SIZE_ * threadIndex;
    milhoja::FArray3D  scratch_base_op1_scratch3D = milhoja::FArray3D{ptr_scratch_base_op1_scratch3D,
            lo_scratch_base_op1_scratch3D,
            hi_scratch_base_op1_scratch3D};

    StaticPhysicsRoutines::computeLaplacianDensity(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    scratch_base_op1_scratch3D,
                    tile_deltas);
}
