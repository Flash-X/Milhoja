#include "errorEstAdv.h"

#include "AmrCoreAdv_F.H"

#include "Grid.h"
#include "Tile.h"
#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

Real Simulation::errorEstAdv(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter) {
    static bool first = true;
    static amrex::Vector<amrex::Real> phierr;
    if(first) {
        first = false;
        //get phierr from parm parse or runtime parameters?
        phierr.push_back(1.01);
        phierr.push_back(1.1);
        phierr.push_back(1.5);
    }

    auto          f   = tileDesc->data();
    const IntVect lo  = tileDesc->lo();
    const IntVect hi  = tileDesc->hi();
    const int     lev = tileDesc->level();

    Real error = 0.0_wp;
    if(lev>=phierr.size()) return error;

    for (    int k=lo.K(); k<=hi.K(); ++k) {
      for (  int j=lo.J(); j<=hi.J(); ++j) {
        for (int i=lo.I(); i<=hi.I(); ++i) {
            if (f(i,j,k,iref) > phierr[lev]) {
                error = 10.0_wp;
            }
    }}}

    return error;

}

