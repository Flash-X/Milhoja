#include "setInitialAdv.h"

#include "AmrCoreAdv_F.H"

#include "Grid.h"
#include "Tile.h"

#include "Flash.h"
#include "constants.h"

void Simulation::setInitialAdv(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();


    int level = tileDesc->level();
    Real time = 0.0_wp;
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();
    Real* dataPtr = tileDesc->dataPtr();
    const IntVect   loGC = tileDesc->loGC();
    const IntVect   hiGC = tileDesc->hiGC();
    const RealVect deltas = tileDesc->deltas();
    const RealVect prob_lo = grid.getProbLo();

    initdata(&level,
             &time,
             AMREX_ARLIM_3D(lo.dataPtr()),
             AMREX_ARLIM_3D(hi.dataPtr()),
             dataPtr,
             AMREX_ARLIM_3D(loGC.dataPtr()),
             AMREX_ARLIM_3D(hiGC.dataPtr()),
             AMREX_ZFILL(deltas.dataPtr()),
             AMREX_ZFILL(prob_lo.dataPtr()) );

}

