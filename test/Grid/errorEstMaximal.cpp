#include "errorEstMaximal.h"

#include "Grid.h"
#include "Tile.h"
#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

//void Simulation::errorEstAdv(const int tId, void* dataItem) {
    //Tile*  tileDesc = static_cast<Tile*>(dataItem);

void Simulation::errorEstMaximal(std::shared_ptr<Tile> tileDesc, int* tptr) {

    const int tagval = amrex::TagBox::SET;
    const IntVect lo = tileDesc->lo();
    const IntVect hi = tileDesc->hi();
    const int tptr_len = (hi-lo+1).product();

    for (int i=0; i<tptr_len; ++i) {
        tptr[i] = tagval;
    }

}

