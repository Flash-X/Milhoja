#include "Tile.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Grid.h"
#include "Grid_IntVect.h"
#include "Flash.h"
#include "constants.h"
using namespace orchestration;

extern "C" {
    void tile_set_limits_fi(Tile* tileDesc, int& gid, int& level,
                            int lo[MDIM],   int hi[MDIM],
                            int loGC[MDIM], int hiGC[MDIM]) {
        gid = static_cast<int>(tileDesc->gridIndex());
        level = static_cast<int>(tileDesc->level());

        const IntVect loPt   = tileDesc->lo();
        const IntVect hiPt   = tileDesc->hi();
        const IntVect loGCPt = tileDesc->loGC();
        const IntVect hiGCPt = tileDesc->hiGC();
        for (unsigned int i=0; i<NDIM; ++i) {
            lo[i]   = loPt[i];
            hi[i]   = hiPt[i];
            loGC[i] = loGCPt[i];
            hiGC[i] = hiGCPt[i];
        }
    }

    void tile_get_data_ptr_fi(Tile* tileDesc, Real*& cptr) {
        cptr = tileDesc->dataPtr();
    }
}

