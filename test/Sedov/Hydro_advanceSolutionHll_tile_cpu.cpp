#include "Hydro.h"

#include <Milhoja.h>
#include <Milhoja_axis.h>
#include <Milhoja_Tile.h>

#include "Sedov.h"
#include "Driver.h"
#include "Eos.h"

void Hydro::advanceSolutionHll_tile_cpu(const int tId,
                                        milhoja::DataItem* dataItem) {
    using namespace milhoja;

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    const IntVect       lo     = tileDesc->lo();
    const IntVect       hi     = tileDesc->hi();
    RealVect            deltas = tileDesc->deltas();
    FArray4D            U      = tileDesc->data();
    FArray4D            flX    = tileDesc->fluxData(Axis::I);
#if MILHOJA_NDIM >= 2
    FArray4D            flY    = tileDesc->fluxData(Axis::J);
#endif
#if MILHOJA_NDIM == 3
    FArray4D            flZ    = tileDesc->fluxData(Axis::K);
#endif

    //----- ALLOCATE SCRATCH MEMORY NEEDED BY PHYSICS
    IntVect    cLo = IntVect{LIST_NDIM(lo.I()-MILHOJA_K1D, lo.J()-MILHOJA_K2D, lo.K()-MILHOJA_K3D)};
    IntVect    cHi = IntVect{LIST_NDIM(hi.I()+MILHOJA_K1D, hi.J()+MILHOJA_K2D, hi.K()+MILHOJA_K3D)};
    FArray3D   auxC = FArray3D::buildScratchArray(cLo, cHi);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block.
    hy::computeFluxesHll(Driver::dt, lo, hi, deltas, U,
                         LIST_NDIM(flX, flY, flZ),
                         auxC);
    hy::updateSolutionHll(lo, hi, U, LIST_NDIM(flX, flY, flZ));
    Eos::idealGammaDensIe(lo, hi, U);
}

