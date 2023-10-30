#include "cpu_tf_fused.h"
#include "Tile_cpu_tf_fused.h"

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

#include "computeLaplacianFused.h"

void  cpu_tf_fused::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_fused*  wrapper = dynamic_cast<Tile_cpu_tf_fused*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    milhoja::FArray4D  CC_1 = tileDesc->data();
    const milhoja::IntVect    lo_base_op1_scratch4D = milhoja::IntVect{LIST_NDIM(tile_lo.I(),
                                       tile_lo.J(),
                                       tile_lo.K())};
    const milhoja::IntVect    hi_base_op1_scratch4D = milhoja::IntVect{LIST_NDIM(tile_hi.I(),
                                       tile_hi.J(),
                                       tile_hi.K())};
    milhoja::Real* ptr_base_op1_scratch4D = 
             static_cast<milhoja::Real*>(Tile_cpu_tf_fused::base_op1_scratch4D_)
            + Tile_cpu_tf_fused::BASE_OP1_SCRATCH4D_SIZE_ * threadIndex;
    milhoja::FArray4D  base_op1_scratch4D = milhoja::FArray4D{ptr_base_op1_scratch4D,
            lo_base_op1_scratch4D,
            hi_base_op1_scratch4D,
            2};

    StaticPhysicsRoutines::computeLaplacianFusedKernels(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    base_op1_scratch4D,
                    tile_deltas);
}
