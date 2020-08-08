#include "errorEstBlank.h"

#include "Grid.h"
#include "Tile.h"

#include "Flash.h"
#include "constants.h"

#include "Grid_AmrCoreFlash.h"

using namespace orchestration;

//void Simulation::errorEstBlank(const int tId, void* dataItem) {
    //Tile*  tileDesc = static_cast<Tile*>(dataItem);

void Simulation::errorEstBlank(int lev, amrex::TagBoxArray& tags, Real time,
                             int ngrow, std::shared_ptr<Tile> tileDesc) {

}

