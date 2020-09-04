#include "errorEstMultiple.h"

#include "Grid.h"
#include "Tile.h"
#include "Flash.h"
#include "constants.h"

#include <AMReX_TagBox.H>

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

void Simulation::errorEstMultiple(std::shared_ptr<Tile> tileDesc, int* tptr) {
    Grid&   grid = Grid::instance();
    int lev = tileDesc->level();
    const int clearval = amrex::TagBox::CLEAR;
    const int tagval = amrex::TagBox::SET;

    static bool first = true;
    static std::vector<Real> ref_value;
    if(first) {
        first = false;
        //get ref_value from parm parse or runtime parameters?
        ref_value.push_back(1.1);
        ref_value.push_back(1.2);
        ref_value.push_back(1.3);
    }
    if(lev>=ref_value.size()) return;

    FArray4D data = tileDesc->data();
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();
    const int tptr_len = (hi-lo+1).product();
    amrex::Box validbox{ amrex::IntVect(lo),
                         amrex::IntVect(hi) };
    amrex::TagBox tagfab{validbox};
    tagfab.setVal(clearval);

    for(int k=lo.K(); k<=hi.K(); ++k) {
    for(int j=lo.J(); j<=hi.J(); ++j) {
    for(int i=lo.I(); i<=hi.I(); ++i) {
        if( data(i,j,k,0)>=ref_value[lev] ) {
            tagfab(amrex::IntVect(LIST_NDIM(i,j,k)),0)
                = tagval;
        }
    }}}

    amrex::Vector<int> itags;
    tagfab.get_itags(itags,validbox);
    for (int i=0; i<tptr_len; ++i) {
        tptr[i] = itags[i];
    }

}

