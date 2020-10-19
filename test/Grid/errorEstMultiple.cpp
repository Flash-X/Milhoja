#include "errorEstMultiple.h"

#include "Flash.h"
#include "constants.h"

#include <vector>

using namespace orchestration;

Real Simulation::errorEstMultiple(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter) {
    Real error = 0.0_wp;

    int lev = tileDesc->level();

    static bool first = true;
    static std::vector<Real> ref_value;
    if(first) {
        first = false;
        ref_value.push_back(1.1);
        ref_value.push_back(1.2);
        ref_value.push_back(1.3);
    }
    if(lev>=ref_value.size()) return error;

    FArray4D data = tileDesc->data();
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();

    for(int k=lo.K(); k<=hi.K(); ++k) {
    for(int j=lo.J(); j<=hi.J(); ++j) {
    for(int i=lo.I(); i<=hi.I(); ++i) {
        if( data(i,j,k,0)>=ref_value[lev] ) {
            error = REFINE_CUTOFF + 1.0_wp;
        }
    }}}

    return error;

}

