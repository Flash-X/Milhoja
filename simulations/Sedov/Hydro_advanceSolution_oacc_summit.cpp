#include "Eos.h"
#include "Hydro.h"
#include "Driver.h"

#include "Tile.h"

#include "Flash.h"

void Hydro::advanceSolution_oacc_summit(const int tId,
                                        orchestration::DataItem* dataItem) {
    using namespace orchestration;

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
    //   * No tiling for now means that these two actions can be fused and that
    //     updateSolution can update U inplace (i.e. Uout = Uin).
    //
    // Compute fluxes
    hy::computeSoundSpeedHll_oacc_summit(lo, hi, U, auxC);
    hy::computeFluxesHll_X_oacc_summit(Driver::dt, lo, hi, deltas, U, flX, auxC);
#if NDIM >= 2
    hy::computeFluxesHll_Y_oacc_summit(Driver::dt, lo, hi, deltas, U, flY, auxC);
#endif
#if NDIM == 3
    hy::computeFluxesHll_Z_oacc_summit(Driver::dt, lo, hi, deltas, U, flZ, auxC);
#endif

    // Update solutions (Uin = Uout)
    hy::scaleSolutionHll_oacc_summit(lo, hi, U, U, flX, flY, flZ);
    hy::updateSolutionHll_FlX_oacc_summit(lo, hi, U, flX);
#if NDIM >= 2
    hy::updateSolutionHll_FlY_oacc_summit(lo, hi, U, flY);
#endif
#if NDIM == 3
    hy::updateSolutionHll_FlZ_oacc_summit(lo, hi, U, flZ);
#endif

    hy::rescaleSolutionHll_oacc_summit(lo, hi, U);
#ifdef EINT_VAR_C
    hy::updateEintHll_oacc_summit(lo, hi, U);
#endif

    // Apply EoS on interior
    Eos::idealGamma_dens_ie(lo, hi, U);
}

