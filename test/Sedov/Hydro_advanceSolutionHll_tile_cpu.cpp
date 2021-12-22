#include "Hydro.h"

#include <Milhoja.h>
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
    FArray4D            U      = tileDesc->data();
    RealVect            deltas = tileDesc->deltas();

    //----- ALLOCATE SCRATCH MEMORY NEEDED BY PHYSICS
    // Fluxes needed on interior and boundary faces only
    IntVect    fHi = IntVect{LIST_NDIM(hi.I()+K1D, hi.J(), hi.K())};
    FArray4D   flX = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

    fHi = IntVect{LIST_NDIM(hi.I(), hi.J()+K2D, hi.K())};
    FArray4D   flY = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

    fHi = IntVect{LIST_NDIM(hi.I(), hi.J(), hi.K()+K3D)};
    FArray4D   flZ = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

    IntVect    cLo = IntVect{LIST_NDIM(lo.I()-K1D, lo.J()-K2D, lo.K()-K3D)};
    IntVect    cHi = IntVect{LIST_NDIM(hi.I()+K1D, hi.J()+K2D, hi.K()+K3D)};
    FArray3D   auxC = FArray3D::buildScratchArray(cLo, cHi);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already
    //   * No tiling for now means that computing fluxes and updating the
    //     solution can be fused and the full advance run independently on each
    //     block.
    hy::computeFluxesHll(Driver::dt, lo, hi, deltas, U, flX, flY, flZ, auxC);
    hy::updateSolutionHll(lo, hi, U, flX, flY, flZ);
    Eos::idealGammaDensIe(lo, hi, U);
}

