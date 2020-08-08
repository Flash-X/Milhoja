#include "errorEstMaximal.h"

#include "Grid.h"
#include "Tile.h"
#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

//void Simulation::errorEstAdv(const int tId, void* dataItem) {
    //Tile*  tileDesc = static_cast<Tile*>(dataItem);

void Simulation::errorEstMaximal(int lev, amrex::TagBoxArray& tags, Real time,
                             int ngrow, std::shared_ptr<Tile> tileDesc) {

    const int tagval = amrex::TagBox::SET;
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();

    static amrex::Vector<int> itags;

    amrex::Box validbox{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::TagBox& tagfab = tags[tileDesc->gridIndex()];
    tagfab.get_itags(itags,validbox);

    for(int i=0;i<itags.size();++i) {
        itags[i] = tagval;
    }

    tagfab.tags_and_untags(itags,validbox);
}

