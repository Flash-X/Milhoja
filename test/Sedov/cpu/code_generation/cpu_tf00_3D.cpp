#include "cpu_tf00_3D.h"
#include "Tile_cpu_tf00_3D.h"

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray2D.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Tile.h>

#include "Eos.h"
#include "Hydro.h"


void  cpu_tf00_3D::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf00_3D*  wrapper = dynamic_cast<Tile_cpu_tf00_3D*>(dataItem);
    Tile*  tileDesc = wrapper->tile_.get();

    Real& dt = wrapper->dt_;
    const IntVect  tile_lo = tileDesc->lo();
    const IntVect  tile_hi = tileDesc->hi();
    const RealVect  tile_deltas = tileDesc->deltas();
    FArray4D  CC_1 = tileDesc->data();
    FArray4D  FLX_1 = tileDesc->fluxData(Axis::I);
    FArray4D  FLY_1 = tileDesc->fluxData(Axis::J);
    FArray4D  FLZ_1 = tileDesc->fluxData(Axis::K);
    IntVect    lo_hydro_op1_auxc = IntVect{LIST_NDIM(tile_lo.I()-MILHOJA_K1D,
                                       tile_lo.J()-MILHOJA_K2D,
                                       tile_lo.K()-MILHOJA_K3D)};
    IntVect    hi_hydro_op1_auxc = IntVect{LIST_NDIM(tile_hi.I()+MILHOJA_K1D,
                                       tile_hi.J()+MILHOJA_K2D,
                                       tile_hi.K()+MILHOJA_K3D)};
    Real* ptr_hydro_op1_auxc = 
             static_cast<Real*>(Tile_cpu_tf00_3D::hydro_op1_auxc_)
            + Tile_cpu_tf00_3D::hydro_op1_auxc_SIZE_ * threadId;
    FArray3D  hydro_op1_auxc = FArray3D{ptr_hydro_op1_auxc,
            lo_hydro_op1_auxc,
            hi_hydro_op1_auxc};

    hy::computeFluxesHll(
                    dt,
                    tile_lo,
                    tile_hi,
                    tile_deltas,
                    CC_1,
                    FLX_1,
                    FLY_1,
                    FLZ_1,
                    hydro_op1_auxc);
    hy::updateSolutionHll(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    FLX_1,
                    FLY_1,
                    FLZ_1);
    Eos::idealGammaDensIe(
                    tile_lo,
                    tile_hi,
                    CC_1);
}