#include "errorEstMaximal.h"

#include <AMReX_TagBox.H>

void Simulation::errorEstMaximal(std::shared_ptr<milhoja::Tile> tileDesc, int* tptr) {
    const int                tagval = amrex::TagBox::SET;
    const milhoja::IntVect   lo = tileDesc->lo();
    const milhoja::IntVect   hi = tileDesc->hi();
    const int                tptr_len = (hi-lo+1).product();

    for (int i=0; i<tptr_len; ++i) {
        tptr[i] = tagval;
    }

}

