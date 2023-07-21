#include "cpu_tf00_2D.h"

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray2D.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Tile.h>

#include "Tile_cpu_tf00_2D.h"

#include "Eos.h"
#include "Hydro.h"

#include "Driver.h"


void  cpu_tf00_2D::taskFunction(const int threadId,
                                milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile_cpu_tf00_2D*  wrapper = dynamic_cast<Tile_cpu_tf00_2D*>(dataItem);
    Tile*  tileDesc = wrapper->tile_.get();

    // Thread-private variables
    Real    dt = wrapper->dt_;

    // Thread-private scratch
    Real*   ptr_hydro_op1_auxc =
                  static_cast<Real*>(Tile_cpu_tf00_2D::auxC_scratch_)
                + Tile_cpu_tf00_2D::AUXC_SIZE_ * threadId;

    const IntVect  tile_lo = tileDesc->lo();
    const IntVect  tile_hi = tileDesc->hi();
    const RealVect  tile_deltas = tileDesc->deltas();
    FArray4D  CC_1 = tileDesc->data();
    FArray4D  FLX_1 = tileDesc->fluxData(Axis::I);
    FArray4D  FLY_1 = tileDesc->fluxData(Axis::J);
    IntVect    cLo = IntVect{LIST_NDIM(tile_lo.I()-MILHOJA_K1D,
                                       tile_lo.J()-MILHOJA_K2D,
                                       tile_lo.K()-MILHOJA_K3D)};
    IntVect    cHi = IntVect{LIST_NDIM(tile_hi.I()+MILHOJA_K1D,
                                       tile_hi.J()+MILHOJA_K2D,
                                       tile_hi.K()+MILHOJA_K3D)};
    FArray3D  hydro_op1_auxc = FArray3D{ptr_hydro_op1_auxc, cLo, cHi};

    hy::computeFluxesHll(
                    dt,
                    tile_lo,
                    tile_hi,
                    tile_deltas,
                    CC_1,
                    FLX_1,
                    FLY_1,
                    hydro_op1_auxc);
    hy::updateSolutionHll(
                    tile_lo,
                    tile_hi,
                    CC_1,
                    FLX_1,
                    FLY_1);
    Eos::idealGammaDensIe(
                    tile_lo,
                    tile_hi,
                    CC_1);
}
