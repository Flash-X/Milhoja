#include "Eos.h"
#include "Hydro.h"
#include "Driver.h"

#include "Tile.h"

#include "Flash.h"

void Hydro::advanceSolutionHll_packet_oacc_summit_2(const int tId,
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
    
    // Compute new solution in scratch block for now
    FArray4D   Unew = FArray4D::buildScratchArray4D(lo, hi, NUNKVAR);

    //----- ADVANCE SOLUTION
    // Update unk data on interiors only
    //   * It is assumed that the GC are filled already.
    // For the present CPU version, we need to have a separate Unew scratch
    // block in which the next solution is computed.  This is due to the fact
    // that the velocities and energies use both the old and new density.
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

    // Update solutions
    hy::updateDensityHll_oacc_summit(lo, hi, U, Unew, flX, flY, flZ);
    hy::updateVelxHll_oacc_summit(lo, hi, U, Unew, flX, flY, flZ);
    hy::updateVelyHll_oacc_summit(lo, hi, U, Unew, flX, flY, flZ);
    hy::updateVelzHll_oacc_summit(lo, hi, U, Unew, flX, flY, flZ);
    hy::updateEnergyHll_oacc_summit(lo, hi, U, Unew, flX, flY, flZ);
    // This won't be necessary once implemented with a data packet with CC1
    // and CC2 data spaces.
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, DENS_VAR_C) = Unew(i, j, k, DENS_VAR_C);
                U(i, j, k, VELX_VAR_C) = Unew(i, j, k, VELX_VAR_C);
                U(i, j, k, VELY_VAR_C) = Unew(i, j, k, VELY_VAR_C);
                U(i, j, k, VELZ_VAR_C) = Unew(i, j, k, VELZ_VAR_C);
                U(i, j, k, ENER_VAR_C) = Unew(i, j, k, ENER_VAR_C);
            }
        }
    }

#ifdef EINT_VAR_C
    hy::computeEintHll_oacc_summit(lo, hi, U);
#endif

    // Apply EoS on interior
    Eos::idealGammaDensIe(lo, hi, U);
}

