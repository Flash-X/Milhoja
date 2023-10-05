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

void  cpu_tf_dens::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_dens*  wrapper = dynamic_cast<Tile_cpu_tf_dens*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    milhoja::FArray4D  CC_1 = tileDesc->data();
    milhoja::IntVect    lo_base_op1_scratch = milhoja::IntVect{LIST_NDIM(tile_lo.I(),
                                       tile_lo.J(),
                                       tile_lo.K())};
    milhoja::IntVect    hi_base_op1_scratch = milhoja::IntVect{LIST_NDIM(tile_hi.I(),
                                       tile_hi.J(),
                                       tile_hi.K())};
    milhoja::Real* ptr_base_op1_scratch = 
             static_cast<milhoja::Real*>(Tile_cpu_tf_dens::base_op1_scratch_)
            + Tile_cpu_tf_dens::BASE_OP1_SCRATCH_SIZE_ * threadId;
    milhoja::FArray3D  base_op1_scratch = milhoja::FArray3D{ptr_base_op1_scratch,
            lo_base_op1_scratch,
            hi_base_op1_scratch};

    StaticPhysicsRoutines::computeLaplacianDensity(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    base_op1_scratch,
                    tile_deltas);
}
