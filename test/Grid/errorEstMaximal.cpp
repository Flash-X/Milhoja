#include "errorEstMaximal.h"

#include "Flash.h"
#include "constants.h"

using namespace orchestration;

Real Simulation::errorEstMaximal(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter) {
    Real refineCutoff = REFINE_CUTOFF;
    return refineCutoff + 1.0_wp;
}

