#include "Tile.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Grid.h"
#include "Flash.h"
#include "constants.h"

extern "C" {
    void tile_set_limits_fi(Tile* tileDesc, int& gid, int& level,
                            int lo[MDIM],   int hi[MDIM],
                            int loGC[MDIM], int hiGC[MDIM]) {
        gid = static_cast<int>(tileDesc->gridIndex());
        level = static_cast<int>(tileDesc->level());

        const int* loPt   = tileDesc->loVect();
        const int* hiPt   = tileDesc->hiVect();
        const int* loGCPt = tileDesc->loGCVect();
        const int* hiGCPt = tileDesc->hiGCVect();
        for (unsigned int i=0; i<AMREX_SPACEDIM; ++i) {
            lo[i]   = loPt[i];
            hi[i]   = hiPt[i];
            loGC[i] = loGCPt[i];
            hiGC[i] = hiGCPt[i];
        }
    }

    void tile_get_data_ptr_fi(Tile* tileDesc,
                              amrex::Real*& cptr) {
        amrex::MultiFab&    unk = Grid::instance().unk();
        amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
        cptr = fab.dataPtr();
    }
}

