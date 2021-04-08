#include <cmath>
#include <algorithm>
#include <iostream>

#include "Simulation.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "FArray2D.h"

#include "constants.h"
#include "Flash.h"
#include "Flash_par.h"

// Hardcoded
const     orchestration::Real   MIN_DIST = 1.0e-10_wp;

/**
  *
  */
void  sim::setInitialConditions(const orchestration::IntVect& lo,
                                const orchestration::IntVect& hi,
                                const unsigned int level,
                                const orchestration::FArray1D& xCoords,
                                const orchestration::FArray1D& yCoords,
                                const orchestration::FArray1D& zCoords,
                                const orchestration::RealVect& deltas,
                                orchestration::FArray4D& solnData) {

}
