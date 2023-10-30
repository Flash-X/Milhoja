#include "cpu_tf_ener.h"
#include "Tile_cpu_tf_ener.h"

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

#include "computeLaplacianEnergy.h"

void  cpu_tf_ener::taskFunction(const int threadIndex,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_ener*  wrapper = dynamic_cast<Tile_cpu_tf_ener*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    milhoja::FArray4D  CC_1 = tileDesc->data();
    const milhoja::IntVect    lo_base_op1_scratch3D = milhoja::IntVect{LIST_NDIM(tile_lo.I(),
                                       tile_lo.J(),
                                       tile_lo.K())};
    const milhoja::IntVect    hi_base_op1_scratch3D = milhoja::IntVect{LIST_NDIM(tile_hi.I(),
                                       tile_hi.J(),
                                       tile_hi.K())};
    milhoja::Real* ptr_base_op1_scratch3D = 
             static_cast<milhoja::Real*>(Tile_cpu_tf_ener::base_op1_scratch3D_)
            + Tile_cpu_tf_ener::BASE_OP1_SCRATCH3D_SIZE_ * threadIndex;
    milhoja::FArray3D  base_op1_scratch3D = milhoja::FArray3D{ptr_base_op1_scratch3D,
            lo_base_op1_scratch3D,
            hi_base_op1_scratch3D};

    StaticPhysicsRoutines::computeLaplacianEnergy(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    base_op1_scratch3D,
                    tile_deltas);
}
