#include "errorEstAdv.h"

#include "AmrCoreAdv_F.H"

#include "Grid.h"
#include "Tile.h"
#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

//void Simulation::errorEstAdv(const int tId, void* dataItem) {
    //Tile*  tileDesc = static_cast<Tile*>(dataItem);

void Simulation::errorEstAdv(int lev, amrex::TagBoxArray& tags, Real time,
                             int ngrow, std::shared_ptr<Tile> tileDesc) {
    Grid&   grid = Grid::instance();

    static bool first = true;
    static amrex::Vector<amrex::Real> phierr;
    if(first) {
        first = false;
        //get phierr from parm parse or runtime parameters?
        phierr.push_back(1.01);
        phierr.push_back(1.1);
        phierr.push_back(1.5);
    }
    if(lev>=phierr.size()) return;

    const int clearval = amrex::TagBox::CLEAR;
    const int tagval = amrex::TagBox::SET;
    const RealVect deltas = tileDesc->deltas();
    const RealVect prob_lo = grid.getProbLo();
    Real* dataPtr = tileDesc->dataPtr();
    const IntVect   loGC = tileDesc->loGC();
    const IntVect   hiGC = tileDesc->hiGC();
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();

    static amrex::Vector<int> itags;

    amrex::Box validbox{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::TagBox& tagfab = tags[tileDesc->gridIndex()];
    tagfab.get_itags(itags,validbox);
    int* tptr = itags.dataPtr();

    state_error(tptr,
                AMREX_ARLIM_3D(lo.dataPtr()),
                AMREX_ARLIM_3D(hi.dataPtr()),
                dataPtr,
                AMREX_ARLIM_3D(loGC.dataPtr()),
                AMREX_ARLIM_3D(hiGC.dataPtr()),
                &tagval,
                &clearval,
                AMREX_ARLIM_3D(lo.dataPtr()),
                AMREX_ARLIM_3D(hi.dataPtr()),
                AMREX_ZFILL(deltas.dataPtr()),
                AMREX_ZFILL(prob_lo.dataPtr()),
                &time,
                &phierr[lev] );

    tagfab.tags_and_untags(itags,validbox);
}

