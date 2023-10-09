#include "cpu_tf_hydro.h"
#include "Tile_cpu_tf_hydro.h"

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

#include "Eos.h"
#include "Hydro.h"

void  cpu_tf_hydro::taskFunction(const int threadId,
                    milhoja::DataItem* dataItem) {
    Tile_cpu_tf_hydro*  wrapper = dynamic_cast<Tile_cpu_tf_hydro*>(dataItem);
    milhoja::Tile*  tileDesc = wrapper->tile_.get();

    const milhoja::Real& dt = wrapper->dt_;
    const milhoja::IntVect  tile_lo = tileDesc->lo();
    const milhoja::IntVect  tile_hi = tileDesc->hi();
    const milhoja::RealVect  tile_deltas = tileDesc->deltas();
    milhoja::FArray4D  CC_1 = tileDesc->data();
    milhoja::FArray4D  FLX_1 = tileDesc->fluxData(milhoja::Axis::I);
    milhoja::FArray4D  FLY_1 = tileDesc->fluxData(milhoja::Axis::J);
    milhoja::FArray4D  FLZ_1 = tileDesc->fluxData(milhoja::Axis::K);
    const milhoja::IntVect    lo_hydro_op1_auxc = milhoja::IntVect{LIST_NDIM(tile_lo.I()-MILHOJA_K1D,
                                       tile_lo.J()-MILHOJA_K2D,
                                       tile_lo.K()-MILHOJA_K3D)};
    const milhoja::IntVect    hi_hydro_op1_auxc = milhoja::IntVect{LIST_NDIM(tile_hi.I()+MILHOJA_K1D,
                                       tile_hi.J()+MILHOJA_K2D,
                                       tile_hi.K()+MILHOJA_K3D)};
    milhoja::Real* ptr_hydro_op1_auxc = 
             static_cast<milhoja::Real*>(Tile_cpu_tf_hydro::hydro_op1_auxc_)
            + Tile_cpu_tf_hydro::HYDRO_OP1_AUXC_SIZE_ * threadId;
    milhoja::FArray3D  hydro_op1_auxc = milhoja::FArray3D{ptr_hydro_op1_auxc,
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