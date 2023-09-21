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

void  cpu_tf_ener::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf_ener*  wrapper = dynamic_cast<Tile_cpu_tf_ener*>(dataItem);
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
             static_cast<milhoja::Real*>(Tile_cpu_tf_ener::base_op1_scratch_)
            + Tile_cpu_tf_ener::BASE_OP1_SCRATCH_SIZE_ * threadId;
    milhoja::FArray4D  base_op1_scratch = milhoja::FArray4D{ptr_base_op1_scratch,
            lo_base_op1_scratch,
            hi_base_op1_scratch,
            2};

    StaticPhysicsRoutines::computeLaplacianEnergy(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    base_op1_scratch,
                    tile_deltas);
}